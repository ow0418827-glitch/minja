"""同一冻结需求的 529 延后补跑账本；不替换需求或重采样有效结果。"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = 1
POLICY_VERSION = "deferred_529_same_demand_v1"


def _fingerprint(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


class Deferred529Evaluation:
    """先遍历原需求，再最多补两轮；原条件的首次有效报告永久保留。

    ``next_attempt`` 返回零基 candidate_index；落盘报告及轨迹索引为一基。
    人工命令入口必须调用 ``start_batch``；同批次标识不能解除停机。
    新标识仅对已经停机的账本重置本批次额度。中断后的未提交请求可以
    再执行，所以外部请求是至少一次语义；不能保证服务端恰好执行一次。
    529 判定由调用方提供，不在此模块解析网络异常。
    """

    def __init__(self, path, trials, conditions, bindings, initial_results=None,
                 is_529=None, primary_count=20, condition_orders=None):
        if isinstance(primary_count, bool) or not isinstance(primary_count, int) or not 1 <= primary_count <= 20:
            raise ValueError("正式需求数必须为 1 至 20 的整数")
        if not isinstance(trials, list) or len(trials) != primary_count:
            raise ValueError("账本只接受固定数量的原正式需求，不接受备用替换需求")
        conditions = list(conditions)
        if (not conditions or len(conditions) > 3 or len(set(conditions)) != len(conditions)
                or any(not isinstance(c, str) or not c for c in conditions)):
            raise ValueError("条件必须为一至三个不同的非空字符串")
        task_ids = [t.get("task_id") if isinstance(t, dict) else None for t in trials]
        if any(not isinstance(t, str) or not t for t in task_ids) or len(set(task_ids)) != len(task_ids):
            raise ValueError("原需求标识必须为不同的非空字符串")
        if is_529 is not None and not callable(is_529):
            raise ValueError("导入的 529 判断器必须可调用")
        orders = ([list(conditions) for _ in trials] if condition_orders is None
                  else copy.deepcopy(condition_orders))
        if isinstance(orders, (list, tuple)):
            orders = [list(order) if isinstance(order, (list, tuple)) else order for order in orders]
        if (not isinstance(orders, list) or len(orders) != primary_count
                or any(not isinstance(order, list) or len(order) != len(conditions)
                       or set(order) != set(conditions) for order in orders)):
            raise ValueError("各需求条件顺序必须为完整且不重复的冻结条件排列")
        self.path, self.trials = Path(path), copy.deepcopy(trials)
        self.conditions, self.primary_count = conditions, primary_count
        self.condition_orders = orders
        self.identity = {"schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION,
            "primary_count": primary_count, "trials_sha256": _fingerprint(trials),
            "conditions": conditions, "condition_orders": orders, "bindings": copy.deepcopy(bindings)}
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state.get("identity") != self.identity:
                raise ValueError("冻结需求、条件或实验绑定已改变")
            self._validate()
        else:
            self.state = {"identity": self.identity, "batch_id": None, "phase": "primary",
                "halt_reason": None, "primary_cursor": 0, "initial_529_streak": 0,
                "rounds_attempted": 0, "round_queue": [], "round_cursor": 0,
                "running": None, "next_attempt_id": 1, "attempts": [], "events": [],
                "slots": [{"task_id": task_id, "slot": index + 1, "initial_done": False,
                    "initial_disposition": None, "results": {}, "latest_529": {},
                    "trajectories": {}} for index, task_id in enumerate(task_ids)]}
        if initial_results is not None:
            self._import(initial_results, is_529 or (lambda report: False))
        self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.state, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            # 提交失败时恢复上一份可读快照，禁止调用方继续把未持久化结果当作已提交。
            if self.path.is_file():
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            raise

    def _slot(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.primary_count:
            raise ValueError("原需求索引越界")
        return self.state["slots"][index]

    def _report(self, index, condition, report):
        slot = self._slot(index)
        if condition not in self.conditions or not isinstance(report, dict):
            raise ValueError("报告必须属于冻结条件")
        saved = copy.deepcopy(report)
        saved.setdefault("task_id", slot["task_id"])
        saved.setdefault("condition", condition)
        if saved["task_id"] != slot["task_id"] or saved["condition"] != condition:
            raise ValueError("报告需求或条件不匹配")
        return saved

    def has_529_history(self, index):
        """真实 529 曾出现过的原需求永久固定；当前条件恢复不会解除。"""
        self._slot(index)
        return any(row["candidate_index"] == index + 1 and row["is_529"]
                   for row in self.state["attempts"])

    def _deferred_after_529(self, index, report):
        flagged = report.get("deferred_after_529", False)
        if not isinstance(flagged, bool):
            raise ValueError("529 后延期标记必须为布尔值")
        if flagged and (not report.get("error") or not self.has_529_history(index)):
            raise ValueError("529 后延期必须有错误且同一原需求已有真实 529 历史")
        return flagged

    def _validate(self):
        state = self.state
        slots = state.get("slots")
        if not isinstance(slots, list) or len(slots) != self.primary_count:
            raise ValueError("账本原需求槽位数量损坏")
        if state.get("phase") not in {"primary", "deferred", "completed"}:
            raise ValueError("账本阶段损坏")
        if state.get("batch_id") is not None and (not isinstance(state["batch_id"], str) or not state["batch_id"]):
            raise ValueError("账本批次标识损坏")
        for name, maximum in (("primary_cursor", self.primary_count), ("rounds_attempted", 2),
                              ("initial_529_streak", self.primary_count)):
            if isinstance(state.get(name), bool) or not isinstance(state.get(name), int) or not 0 <= state[name] <= maximum:
                raise ValueError("账本计数器损坏：" + name)
        attempts = state.get("attempts")
        if not isinstance(attempts, list) or not isinstance(state.get("events"), list):
            raise ValueError("账本审计记录损坏")
        attempt_ids, prior_529_slots = set(), set()
        for row in attempts:
            index = row.get("candidate_index", 0) - 1
            saved = self._report(index, row.get("condition"), row.get("report"))
            identifier = row.get("attempt_id")
            if (isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 1
                    or identifier in attempt_ids or saved != row["report"]
                    or not isinstance(row.get("is_529"), bool)
                    or row["is_529"] and not saved.get("error")
                    or row.get("phase") not in {"primary", "deferred", "imported"}
                    or isinstance(row.get("round"), bool) or row.get("round") not in (0, 1, 2)
                    or row.get("trajectory") is not None and not isinstance(row["trajectory"], dict)):
                raise ValueError("账本已提交请求损坏")
            attempt_ids.add(identifier)
            flagged = saved.get("deferred_after_529", False)
            if (not isinstance(flagged, bool)
                    or flagged and (not saved.get("error") or index not in prior_529_slots)):
                raise ValueError("529 后延期审计必须具有先前真实 529 和当前错误")
            if row["is_529"]:
                prior_529_slots.add(index)
        if (isinstance(state.get("next_attempt_id"), bool) or not isinstance(state.get("next_attempt_id"), int)
                or state["next_attempt_id"] <= max(attempt_ids, default=0)):
            raise ValueError("账本请求序号损坏")
        for index, slot in enumerate(slots):
            if (slot.get("task_id") != self.trials[index]["task_id"] or slot.get("slot") != index + 1
                    or not isinstance(slot.get("initial_done"), bool)
                    or slot.get("initial_disposition") not in {None, "529", "committed", "other_technical"}
                    or slot["initial_done"] != (slot["initial_disposition"] is not None)):
                raise ValueError("账本原需求身份损坏")
            for field in ("results", "latest_529", "trajectories"):
                if not isinstance(slot.get(field), dict) or set(slot[field]) - set(self.conditions):
                    raise ValueError("账本条件结果损坏")
            if set(slot["results"]) & set(slot["latest_529"]):
                raise ValueError("有效结果与延期错误不能同时占用同一条件")
            if (slot["initial_disposition"] == "other_technical"
                    and (self.has_529_history(index) or slot["latest_529"]
                         or not any(r.get("error") for r in slot["results"].values()))):
                raise ValueError("交备用的槽位必须具有技术错误且从未出现真实 529")
            for field in ("results", "latest_529"):
                for condition, report in slot[field].items():
                    self._report(index, condition, report)
                    matching = [a for a in attempts if a["candidate_index"] == index + 1
                                and a["condition"] == condition and a["report"] == report
                                and (a["is_529"] or a["report"].get("deferred_after_529", False))
                                    == (field == "latest_529")]
                    if not matching:
                        raise ValueError("账本结果缺少已提交审计记录")
        queue = state.get("round_queue")
        cursor = state.get("round_cursor")
        if (not isinstance(queue, list) or isinstance(cursor, bool) or not isinstance(cursor, int)
                or not 0 <= cursor <= len(queue)):
            raise ValueError("补跑队列损坏")
        for entry in queue:
            self._slot(entry.get("candidate_index"))
            if entry.get("condition") not in self.conditions:
                raise ValueError("补跑条件损坏")
        running = state.get("running")
        if running is not None:
            self._slot(running.get("candidate_index"))
            if (running.get("condition") not in self.conditions or running.get("phase") != state["phase"]
                    or running.get("attempt_id") != state["next_attempt_id"]
                    or running.get("round") != (state["rounds_attempted"] if state["phase"] == "deferred" else 0)):
                raise ValueError("未提交请求的续跑信息损坏")

    def _archive(self, index, condition, report, trajectory, is_529, phase, round_number, *, imported=False):
        identifier = self.state["next_attempt_id"]
        self.state["next_attempt_id"] += 1
        row = {"attempt_id": identifier, "candidate_index": index + 1, "evaluation_slot": index + 1,
               "attempt_number": 1 + sum(a["candidate_index"] == index + 1 and a["condition"] == condition
                                         for a in self.state["attempts"]),
               "task_id": self._slot(index)["task_id"], "condition": condition,
               "phase": phase, "round": round_number, "batch_id": self.state["batch_id"],
               "is_529": is_529, "imported": imported, "committed_at": _now(),
               "report": copy.deepcopy(report), "trajectory": copy.deepcopy(trajectory)}
        self.state["attempts"].append(row)
        return row

    def _import(self, initial_results, detector):
        if isinstance(initial_results, dict):
            if set(initial_results) - set(self.conditions):
                raise ValueError("导入结果包含未知条件")
            rows = [(condition, row) for condition, reports in initial_results.items() for row in reports]
        elif isinstance(initial_results, list):
            rows = [(row.get("condition"), row) for row in initial_results]
        else:
            raise ValueError("导入结果必须为按条件分组的列表或报告列表")
        # 条件分组可能把后续混合错误放在原 529 之前；先导入非延期报告，
        # 再核验延期报告确有同一需求的真实 529 来源。首次有效结果仍优先。
        rows.sort(key=lambda item: bool(item[1].get("report", item[1]).get("deferred_after_529", False))
                  if isinstance(item[1], dict) and isinstance(item[1].get("report", item[1]), dict) else False)
        indexes = {slot["task_id"]: index for index, slot in enumerate(self.state["slots"])}
        imported_529 = set()
        for condition, row in rows:
            if not isinstance(row, dict):
                raise ValueError("导入报告必须为对象")
            report = row.get("report", row)
            trajectory = row.get("trajectory") if "report" in row else None
            if not isinstance(report, dict) or report.get("task_id") not in indexes:
                raise ValueError("导入报告必须用原需求标识绑定，不能按列表位置猜测")
            index = indexes[report["task_id"]]
            condition = report.get("condition", condition)
            saved = self._report(index, condition, report)
            slot = self._slot(index)
            overload = bool(detector(saved))
            if overload and not saved.get("error"):
                raise ValueError("529 导入必须具有错误报告")
            deferred = self._deferred_after_529(index, saved)
            pending_error = overload or deferred
            previous = slot["results"].get(condition)
            if previous is not None:
                if previous != saved:
                    # 已提交有效结果优先于历史 529；正常结果不能被另一正常报告替换。
                    if not pending_error:
                        raise ValueError("已提交条件报告不能被导入覆盖")
                    if not any(a["candidate_index"] == index + 1 and a["condition"] == condition
                               and a["report"] == saved for a in self.state["attempts"]):
                        self._archive(index, condition, saved, trajectory, overload, "imported", 0, imported=True)
                continue
            if any(a["candidate_index"] == index + 1 and a["condition"] == condition
                   and a["report"] == saved for a in self.state["attempts"]):
                continue
            self._archive(index, condition, saved, trajectory, overload, "imported", 0, imported=True)
            if pending_error:
                slot["latest_529"][condition] = saved
                imported_529.add(index)
            else:
                slot["results"][condition] = saved
                slot["latest_529"].pop(condition, None)
                if trajectory is not None:
                    slot["trajectories"][condition] = copy.deepcopy(trajectory)
        for index, slot in enumerate(self.state["slots"]):
            if slot["initial_disposition"] == "other_technical":
                continue
            if index in imported_529 or slot["latest_529"]:
                slot["initial_done"], slot["initial_disposition"] = True, "529"
            elif len(slot["results"]) == len(self.conditions):
                slot["initial_done"], slot["initial_disposition"] = True, "committed"

    def start_batch(self, batch_id):
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("人工启动批次标识必须为非空字符串")
        if self.state["batch_id"] == batch_id:
            return self.audit()
        old_batch = self.state["batch_id"]
        reset = self.state["halt_reason"] is not None
        self.state["events"].append({"event": "start" if old_batch is None else "resume",
            "at": _now(), "previous_batch_id": old_batch, "batch_id": batch_id,
            "previous_halt_reason": self.state["halt_reason"],
            "previous_rounds_attempted": self.state["rounds_attempted"],
            "previous_initial_529_streak": self.state["initial_529_streak"],
            "new_retry_allowance": reset})
        self.state["batch_id"] = batch_id
        if reset:
            self.state["halt_reason"] = None
            self.state["initial_529_streak"] = 0
            self.state["rounds_attempted"] = 0
            self.state["round_queue"], self.state["round_cursor"] = [], 0
            self.state["phase"] = "primary" if not self.completed_initial_pass else "deferred"
        self._save()
        return self.audit()

    @property
    def completed_initial_pass(self):
        return all(slot["initial_done"] for slot in self.state["slots"])

    def _pending_pairs(self):
        return [{"candidate_index": index, "condition": condition}
                for index, slot in enumerate(self.state["slots"])
                for condition in self.condition_orders[index]
                if slot["initial_disposition"] != "other_technical" and condition not in slot["results"]]

    def _next(self):
        if self.state["halt_reason"]:
            return None
        if self.state["running"] is not None:
            return copy.deepcopy(self.state["running"])
        while True:
            if self.state["phase"] == "primary":
                cursor = self.state["primary_cursor"]
                if cursor < self.primary_count:
                    slot = self._slot(cursor)
                    if not slot["initial_done"]:
                        for condition in self.condition_orders[cursor]:
                            if condition not in slot["results"]:
                                return {"candidate_index": cursor, "slot": cursor + 1,
                                    "task_id": slot["task_id"], "condition": condition,
                                    "phase": "primary", "round": 0}
                        slot["initial_done"], slot["initial_disposition"] = True, "committed"
                    if slot["initial_disposition"] in {"committed", "other_technical"}:
                        self.state["initial_529_streak"] = 0
                    self.state["primary_cursor"] += 1
                    continue
                self.state["phase"] = "deferred"
            if self.state["phase"] == "deferred":
                queue, cursor = self.state["round_queue"], self.state["round_cursor"]
                while cursor < len(queue) and (
                        queue[cursor]["condition"] in self._slot(queue[cursor]["candidate_index"])["results"]
                        or self._slot(queue[cursor]["candidate_index"])["initial_disposition"] == "other_technical"):
                    cursor += 1
                    self.state["round_cursor"] = cursor
                if cursor < len(queue):
                    entry = queue[cursor]
                    slot = self._slot(entry["candidate_index"])
                    return {**entry, "slot": slot["slot"], "task_id": slot["task_id"],
                            "phase": "deferred", "round": self.state["rounds_attempted"]}
                pending = self._pending_pairs()
                if not pending:
                    self.state["phase"] = "completed"
                elif self.state["rounds_attempted"] >= 2:
                    self.state["halt_reason"] = "deferred_rounds_exhausted"
                    return None
                else:
                    self.state["rounds_attempted"] += 1
                    self.state["round_queue"], self.state["round_cursor"] = pending, 0
                    continue
            return None

    def next_attempt(self):
        if self.state["batch_id"] is None:
            raise ValueError("人工命令入口需先登记启动批次")
        result = self._next()
        self._save()
        return result

    def begin(self, index, condition):
        planned = self.next_attempt()
        if planned is None or planned["candidate_index"] != index or planned["condition"] != condition:
            raise ValueError("请求不是当前已冻结的下一个补跑任务")
        if self.state["running"] is None:
            self.state["running"] = {**planned, "attempt_id": self.state["next_attempt_id"],
                                     "batch_id": self.state["batch_id"], "started_at": _now()}
            self._save()
        return copy.deepcopy(self.state["running"])

    def record_result(self, index, condition, report, *, trajectory=None, is_529=False):
        if not isinstance(is_529, bool) or trajectory is not None and not isinstance(trajectory, dict):
            raise ValueError("529 标记必须为布尔值，轨迹必须为对象")
        saved = self._report(index, condition, report)
        slot = self._slot(index)
        deferred = self._deferred_after_529(index, saved)
        pending_error = is_529 or deferred
        previous = slot["results"].get(condition)
        if previous is not None:
            if previous != saved or trajectory is not None and slot["trajectories"].get(condition) != trajectory:
                raise ValueError("首次已提交结果及轨迹不可覆盖")
            return copy.deepcopy(slot)
        if is_529 and not saved.get("error"):
            raise ValueError("529 结果必须具有错误报告")
        running = self.state["running"]
        if running is None or running["candidate_index"] != index or running["condition"] != condition:
            if (running is None and pending_error and self.state["attempts"]
                    and self.state["attempts"][-1]["candidate_index"] == index + 1
                    and self.state["attempts"][-1]["condition"] == condition
                    and self.state["attempts"][-1]["is_529"] == is_529
                    and self.state["attempts"][-1]["report"] == saved
                    and self.state["attempts"][-1]["trajectory"] == trajectory):
                return copy.deepcopy(slot)
            raise ValueError("提交结果前必须登记当前正在执行的请求")
        self._archive(index, condition, saved, trajectory, is_529, running["phase"], running["round"])
        if pending_error:
            # 内部字段兼容旧账本名；报告及审计的 is_529 保留真实错误分类。
            slot["latest_529"][condition] = saved
            if running["phase"] == "primary":
                slot["initial_done"], slot["initial_disposition"] = True, "529"
                if is_529:
                    self.state["initial_529_streak"] += 1
                    if self.state["initial_529_streak"] >= 3:
                        self.state["halt_reason"] = "three_consecutive_primary_529"
        else:
            slot["results"][condition] = saved
            slot["latest_529"].pop(condition, None)
            if trajectory is not None:
                slot["trajectories"][condition] = copy.deepcopy(trajectory)
            if running["phase"] == "primary" and len(slot["results"]) == len(self.conditions):
                slot["initial_done"], slot["initial_disposition"] = True, "committed"
                self.state["initial_529_streak"] = 0
        if running["phase"] == "deferred":
            self.state["round_cursor"] += 1
        self.state["running"] = None
        self._next()
        self._save()
        return copy.deepcopy(slot)

    def exclude_other_error(self, index):
        """仅供既有备用机制提前排除非 529 技术错误槽位；不伪造余下结果。"""
        slot = self._slot(index)
        if self.has_529_history(index):
            raise ValueError("曾出现真实 529 的原需求不得交备用替换")
        if slot["latest_529"] or not any(report.get("error") for report in slot["results"].values()):
            raise ValueError("只有没有待补 529 的非 529 技术错误槽位可交备用")
        if self.state["running"] is not None and self.state["running"]["candidate_index"] == index:
            raise ValueError("不能排除尚未提交的正在执行请求")
        if slot["initial_disposition"] != "other_technical":
            slot["initial_done"], slot["initial_disposition"] = True, "other_technical"
            self.state["events"].append({"event": "exclude_other_technical", "at": _now(),
                                        "batch_id": self.state["batch_id"], "evaluation_slot": index + 1})
            self._next()
            self._save()
        return copy.deepcopy(slot)

    def final_results(self):
        result = {condition: [] for condition in self.conditions}
        for slot in self.state["slots"]:
            for condition in self.conditions:
                saved = slot["results"].get(condition, slot["latest_529"].get(condition))
                if saved is not None:
                    report = copy.deepcopy(saved)
                    report.setdefault("evaluation_slot", slot["slot"])
                    report.setdefault("candidate_index", slot["slot"])
                    report.setdefault("candidate_role", "primary")
                    report.setdefault("replaces_task_id", None)
                    result[condition].append(report)
        return result

    def saved_trajectories(self):
        return [{k: copy.deepcopy(row[k]) for k in ("candidate_index", "evaluation_slot", "task_id",
                "condition", "attempt_id", "attempt_number", "phase", "round", "batch_id", "is_529", "trajectory")}
                for row in self.state["attempts"] if row["trajectory"] is not None]

    def error_events(self):
        return [copy.deepcopy(row) for row in self.state["attempts"] if row["report"].get("error")]

    @property
    def complete(self):
        return self.audit()["status"] == "completed"

    def audit(self):
        pending = self._pending_pairs()
        missing_slots = sorted({pair["candidate_index"] + 1 for pair in pending})
        status = ("halted" if self.state["halt_reason"] else "completed" if not pending
                  else "deferred" if self.completed_initial_pass else "pending")
        other = sum(bool(report.get("error")) for slot in self.state["slots"] for report in slot["results"].values())
        return {"schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION,
            "primary_count": self.primary_count, "candidate_pool_size": self.primary_count,
            "status": status, "halt_reason": self.state["halt_reason"], "batch_id": self.state["batch_id"],
            "completed_initial_pass": self.completed_initial_pass,
            "rounds_attempted": self.state["rounds_attempted"],
            "initial_529_streak": self.state["initial_529_streak"],
            "pending_slots": missing_slots, "unfilled_slots": missing_slots,
            "unfilled_count": len(missing_slots), "pending_conditions": copy.deepcopy(pending),
            "accepted_pairs": sum(len(slot["results"]) == len(self.conditions)
                                  and not any(r.get("error") for r in slot["results"].values())
                                  for slot in self.state["slots"]),
            "attempted_pairs": sum(slot["initial_done"] or bool(slot["results"]) or bool(slot["latest_529"])
                                   for slot in self.state["slots"]),
            "committed_attempts": len(self.state["attempts"]),
            "overload_529_errors": sum(row["is_529"] for row in self.state["attempts"]),
            "deferred_after_529_errors": sum(row["report"].get("deferred_after_529", False)
                                              for row in self.state["attempts"]),
            "other_technical_errors": other,
            "other_excluded_slots": [slot["slot"] for slot in self.state["slots"]
                                     if slot["initial_disposition"] == "other_technical"],
            "technical_errors_total": sum(bool(row["report"].get("error")) for row in self.state["attempts"]),
            "running": copy.deepcopy(self.state["running"])}
