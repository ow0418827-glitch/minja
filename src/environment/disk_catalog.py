"""Read-only, bounded-memory product access and streaming catalog preparation.

Serving an existing RAP catalog never reads products.jsonl or the raw JSON array.
Full preparation writes only below this project's output directories. Lucene is
the default. SQLite FTS5 is an explicitly selected alternative whose different
ranking is recorded and must not be mixed with Lucene baselines.
"""
from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class ProductMap(Mapping):
    def __init__(self, catalog):
        self.catalog = catalog

    def __getitem__(self, key):
        return self.catalog.product(key)

    def __contains__(self, key):
        return self.catalog.one("SELECT 1 FROM products WHERE asin=?", (key,)) is not None

    def __iter__(self):
        return (row[0] for row in self.catalog.rows("SELECT asin FROM products ORDER BY ordinal"))

    def __len__(self):
        return self.catalog.count


class PriceMap(ProductMap):
    def __getitem__(self, key):
        row = self.catalog.one("SELECT price FROM products WHERE asin=?", (key,))
        if row is None:
            raise KeyError(key)
        return row[0]


class AttributeMap(Mapping):
    def __init__(self, catalog):
        self.catalog = catalog

    def __getitem__(self, key):
        # Compatibility for source callers; special attribute search below does
        # not materialize this potentially large set.
        rows = self.catalog.rows("SELECT asin FROM attribute_products WHERE attribute=?", (key,))
        return {row[0] for row in rows}

    def __contains__(self, key):
        return self.catalog.one("SELECT 1 FROM attribute_products WHERE attribute=? LIMIT 1", (key,)) is not None

    def __iter__(self):
        return (row[0] for row in self.catalog.rows("SELECT DISTINCT attribute FROM attribute_products"))

    def __len__(self):
        return self.catalog.one("SELECT count(DISTINCT attribute) FROM attribute_products")[0]


class ProductSequence(Sequence):
    """SQL-backed sequence: pagination fetches only the requested page."""
    def __init__(self, catalog, where="", parameters=()):
        self.catalog, self.where, self.parameters = catalog, where, tuple(parameters)
        self._count = None

    def __len__(self):
        if self._count is None:
            self._count = self.catalog.one("SELECT count(*) FROM products p " + self.where, self.parameters)[0]
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            rows = self.catalog.rows("SELECT data FROM products p " + self.where +
                                     " ORDER BY ordinal LIMIT ? OFFSET ?",
                                     self.parameters + (max(0, stop - start), start))
            return [self.catalog.decode(row[0]) for row in rows]
        if not isinstance(index, int):
            raise TypeError("Product index must be an integer or slice")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Avoid OFFSET over hundreds of thousands of rows for unfiltered access.
        if not self.where:
            row = self.catalog.one("SELECT data FROM products WHERE ordinal=?", (index,))
        else:
            row = self.catalog.one("SELECT data FROM products p " + self.where +
                                   " ORDER BY ordinal LIMIT 1 OFFSET ?", self.parameters + (index,))
        if row is None:
            raise IndexError(index)
        return self.catalog.decode(row[0])

    def __iter__(self):
        return (self.catalog.decode(row[0]) for row in self.catalog.rows(
            "SELECT data FROM products p " + self.where + " ORDER BY ordinal", self.parameters))


class DiskCatalog:
    def __init__(self, root, cache_mb=16):
        self.root = Path(root).resolve()
        self.data_manifest = read_json(self.root / "data_manifest.json")
        self.index_manifest = read_json(self.root / "index_manifest.json")
        if self.index_manifest["bundle_id"] != self.data_manifest["bundle_id"]:
            raise ValueError("Index/catalog bundle mismatch")
        self.goals = read_json(self.root / "goals.json")
        if fingerprint(self.goals) != self.data_manifest["goals_sha256"]:
            raise ValueError("Goal data changed")
        if not self.goals:
            raise ValueError("Catalog has no usable shopping goals")
        database = self.root / "catalog.sqlite"
        if not database.is_file():
            raise FileNotFoundError(database)
        self.conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        self.lock = threading.RLock()
        self.decoded_products = 0
        self.conn.execute("PRAGMA query_only=ON")
        self.conn.execute("PRAGMA mmap_size=0")
        self.conn.execute(f"PRAGMA cache_size={-max(1, int(cache_mb)) * 1024}")
        try:
            self.columns = {row[1] for row in self.conn.execute("PRAGMA table_info(products)")}
            if not {"asin", "ordinal", "data", "price"}.issubset(self.columns):
                raise ValueError("Unsupported product database schema")
            self.count = self.one("SELECT count(*) FROM products")[0]
            if self.count != self.data_manifest["products"]:
                raise ValueError("Catalog product count mismatch")
            self.products, self.prices = ProductMap(self), PriceMap(self)
            self.attributes, self.sequence = AttributeMap(self), ProductSequence(self)
            self.backend = self.index_manifest.get("backend", "lucene")
            if self.backend not in {"lucene", "sqlite_fts5"}:
                raise ValueError("Unsupported search backend: " + self.backend)
            if self.index_manifest["documents"] + len(self.index_manifest.get("empty_product_asins", [])) != self.count:
                raise ValueError("Search index/catalog coverage mismatch")
            if self.backend == "sqlite_fts5":
                docs = self.one("SELECT count(*) FROM products_fts")[0]
                if docs != self.index_manifest["documents"]:
                    raise ValueError("FTS/catalog document count mismatch")
        except Exception:
            self.conn.close()
            raise

    @classmethod
    def open(cls, root, cache_mb=16):
        return cls(root, cache_mb)

    def one(self, sql, parameters=()):
        with self.lock:
            return self.conn.execute(sql, parameters).fetchone()

    def rows(self, sql, parameters=()):
        with self.lock:
            cursor = self.conn.execute(sql, parameters)
        try:
            while True:
                with self.lock:
                    batch = cursor.fetchmany(64)
                if not batch:
                    break
                yield from batch
        finally:
            with self.lock:
                cursor.close()

    def decode(self, value):
        self.decoded_products += 1
        return json.loads(value)

    def product(self, asin):
        row = self.one("SELECT data FROM products WHERE asin=?", (asin,))
        if row is None:
            raise KeyError(asin)
        return self.decode(row[0])

    def search(self, keywords, search_engine=None, k=50):
        if not keywords:
            return []
        if keywords[0] == "<r>":
            return random.sample(self.sequence, k=min(k, self.count))
        if keywords[0] == "<a>":
            value = " ".join(keywords[1:]).strip()
            return ProductSequence(self, "WHERE p.asin IN (SELECT asin FROM attribute_products WHERE attribute=?)", (value,))
        if keywords[0] in {"<c>", "<q>"}:
            field = "category" if keywords[0] == "<c>" else "query"
            value = keywords[1].strip() if field == "category" and len(keywords) > 1 else " ".join(keywords[1:]).strip()
            expression = "p." + field if field in self.columns else "json_extract(p.data, '$." + field + "')"
            return ProductSequence(self, "WHERE " + expression + "=?", (value,))
        query = " ".join(keywords)
        if self.backend == "lucene":
            if search_engine is None:
                raise ValueError("Lucene searcher is required for this catalog")
            asins = [json.loads(search_engine.doc(hit.docid).raw())["id"]
                     for hit in search_engine.search(query, k=k)]
        else:
            # Escape all FTS operators supplied as natural query text.
            tokens = re.findall(r"\w+", query, flags=re.UNICODE)
            if not tokens:
                return []
            match = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
            asins = [row[0] for row in self.rows(
                "SELECT asin FROM products_fts WHERE products_fts MATCH ? ORDER BY bm25(products_fts), rowid LIMIT ?", (match, k))]
        return [self.products[asin] for asin in asins if asin in self.products]

    def diagnostics(self):
        return {"disk_backed": True, "database_read_only": True,
                "scope": self.data_manifest.get("scope", "unspecified"),
                "products": self.count, "decoded_products": self.decoded_products,
                "search_backend": self.backend, "bundle_id": self.data_manifest["bundle_id"],
                "goals_sha256": self.data_manifest["goals_sha256"],
                "database": str(self.root / "catalog.sqlite"), "mmap_bytes": 0,
                "sqlite_cache_kib": -self.one("PRAGMA cache_size")[0]}

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def normalized(product, attributes, human):
    """The existing catalog_3gb/upstream human-goal field conversions."""
    p = dict(product)
    for key in ("product_information", "brand", "brand_url", "list_price", "availability_quantity",
                "availability_status", "total_reviews", "total_answered_questions", "seller_id",
                "seller_name", "fulfilled_by_amazon", "fast_track_message", "aplus_present", "small_description_old"):
        p.pop(key, None)
    p["name"] = p.get("name") or ""
    p["full_description"] = p.get("full_description") or ""
    p["Title"], p["Description"] = p["name"], p["full_description"]
    p["Reviews"], p["Rating"] = [], "N.A."
    description = p.get("small_description")
    p["BulletPoints"] = description if isinstance(description, list) else [description or ""]
    pricing = p.get("pricing")
    prices = [100.0] if not pricing else [float(Decimal(re.sub(r"[^\d.]", "", v))) for v in pricing.split("$")[1:]][:2]
    prices = prices or [100.0]
    p["pricing"] = prices
    p["Price"] = f"${prices[0]}" if len(prices) == 1 else f"${prices[0]} to ${prices[1]}"
    options, images = {}, {}
    for name, contents in (p.get("customization_options") or {}).items():
        if contents is None:
            continue
        values = []
        for entry in contents:
            value = entry["value"].strip().replace("/", " | ").lower()
            values.append(value)
            images[value] = entry.get("image")
        options[name.lower()] = values
    p["options"], p["option_to_image"] = options, images
    p["Attributes"] = attributes.get("attributes", ["DUMMY_ATTR"])
    if p["Attributes"] is None:
        p["Attributes"] = ["DUMMY_ATTR"]
    if human is not None:
        p["instructions"] = human
    p["images"] = p.get("images") or []
    p["MainImage"] = p["images"][0] if p["images"] else ""
    p["query"] = (p.get("query") or "").lower().strip()
    return p


def search_contents(product):
    options = ", and ".join(f"{name}: {', '.join(values)}" for name, values in product["options"].items())
    bullet = product["BulletPoints"][0] if product["BulletPoints"] else ""
    return " ".join([product["Title"] or "", product["Description"] or "", bullet or "", options]).lower()


def _output_destination(destination):
    destination = Path(destination).resolve()
    allowed = (PROJECT_ROOT / "src" / "output", PROJECT_ROOT / "output")
    if not any(destination.is_relative_to(path) and destination != path for path in allowed):
        raise ValueError("New catalogs must be written below the current minja output directory")
    if destination.exists():
        raise FileExistsError("Choose an unused catalog destination: " + str(destination))
    return destination


def build_catalog(raw_items, raw_attributes, human_goals, destination, webshop_source,
                  seed=20260928, index_path=None, goal_builder=None, progress=None,
                  search_backend="lucene", targets=None):
    """Stream the entire source JSON array; no product/attribute dictionary load.

    Interrupted builds remain under a uniquely named current-project staging
    directory and are never published as a complete catalog.
    """
    import ijson
    if search_backend not in {"lucene", "sqlite_fts5"}:
        raise ValueError("Choose the explicit lucene or sqlite_fts5 search backend")
    if index_path is not None and search_backend != "lucene":
        raise ValueError("An existing Lucene index requires the lucene backend")
    destination = _output_destination(destination)
    raw_items, raw_attributes, human_goals = map(lambda p: Path(p).resolve(),
                                               (raw_items, raw_attributes, human_goals))
    source_paths = {"items": raw_items, "attributes": raw_attributes, "human_goals": human_goals}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if index_path is not None and not Path(index_path).resolve().is_dir():
        raise FileNotFoundError(index_path)
    if index_path is not None and not any(Path(index_path).glob("segments_*")):
        raise ValueError("Provided Lucene index is empty or incomplete")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (destination.name + ".building_" + uuid.uuid4().hex[:10])
    staging.mkdir()
    database = staging / "catalog.sqlite"
    conn = sqlite3.connect(database)
    conn.execute("PRAGMA cache_size=-16384")
    conn.execute("PRAGMA mmap_size=0")
    conn.executescript("""CREATE TABLE attributes(asin TEXT PRIMARY KEY, data TEXT);
        CREATE TABLE products(asin TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, data TEXT,
                              price REAL, category TEXT, query TEXT);
        CREATE TABLE attribute_products(attribute TEXT, asin TEXT);
        CREATE INDEX attribute_lookup ON attribute_products(attribute);
        CREATE INDEX product_category_lookup ON products(category);
        CREATE INDEX product_query_lookup ON products(query);""")
    use_fts = search_backend == "sqlite_fts5"
    if use_fts:
        conn.execute("CREATE VIRTUAL TABLE products_fts USING fts5(asin UNINDEXED, contents)")
    rng = random.Random(seed)
    human = read_json(human_goals)  # Small human task file, never the product array.
    target_entries = read_json(targets) if isinstance(targets, (str, Path)) else targets
    docs_stream = None
    if not use_fts and index_path is None:
        (staging / "search_documents").mkdir()
        docs_stream = (staging / "search_documents" / "documents.jsonl").open("w", encoding="utf-8")
    count, seen, skipped, documents = 0, 0, 0, 0
    empty_asins = []
    normalization_stats = {"empty_images": 0, "empty_title": 0, "empty_bullets": 0, "default_price": 0}
    try:
        with raw_attributes.open("rb") as stream:
            batch = []
            for asin, entry in ijson.kvitems(stream, "", use_float=True):
                batch.append((asin, json.dumps(entry, ensure_ascii=False)))
                if len(batch) == 2000:
                    conn.executemany("INSERT INTO attributes VALUES (?,?)", batch)
                    conn.commit()
                    batch = []
            conn.executemany("INSERT INTO attributes VALUES (?,?)", batch)
            conn.commit()
        if progress is not None:
            progress({"phase": "attributes_complete"})
        with raw_items.open("rb") as stream:
            for raw in ijson.items(stream, "item", use_float=True):
                seen += 1
                asin = raw["asin"]
                if asin == "nan" or len(asin) > 10 or conn.execute("SELECT 1 FROM products WHERE asin=?", (asin,)).fetchone():
                    skipped += 1
                    continue
                attr = conn.execute("SELECT data FROM attributes WHERE asin=?", (asin,)).fetchone()
                product = normalized(raw, json.loads(attr[0]) if attr else {}, human.get(asin))
                normalization_stats["empty_images"] += not bool(raw.get("images"))
                normalization_stats["empty_title"] += not bool(raw.get("name"))
                normalization_stats["empty_bullets"] += not bool(raw.get("small_description"))
                normalization_stats["default_price"] += not bool(raw.get("pricing"))
                prices = product["pricing"]
                price = prices[0] if len(prices) == 1 else rng.uniform(*prices[:2])
                conn.execute("INSERT INTO products VALUES (?,?,?,?,?,?)", (asin, count,
                    json.dumps(product, ensure_ascii=False), price, product["category"], product["query"]))
                conn.executemany("INSERT INTO attribute_products VALUES (?,?)", ((a, asin) for a in set(product["Attributes"])))
                contents = search_contents(product)
                if docs_stream is not None:
                    docs_stream.write(json.dumps({"id": asin, "contents": contents}, ensure_ascii=False) + "\n")
                if contents.strip():
                    documents += 1
                    if use_fts:
                        conn.execute("INSERT INTO products_fts(asin,contents) VALUES (?,?)", (asin, contents))
                else:
                    empty_asins.append(asin)
                count += 1
                if count % 2000 == 0:
                    conn.commit()
                    if progress is not None:
                        progress({"phase": "products", "products": count, "source_records_seen": seen,
                                  "source_bytes_read": stream.tell()})
        conn.commit()
        if docs_stream is not None:
            docs_stream.close()
            docs_stream = None
        if target_entries is not None:
            for pair in target_entries:
                if not pair.get("target_asins"):
                    raise ValueError("Target mapping has no product identifiers")
                expected = re.sub(r"\s+", "", pair["target"]).casefold()
                for asin in pair["target_asins"]:
                    row = conn.execute("SELECT data FROM products WHERE asin=?", (asin,)).fetchone()
                    if row is None:
                        raise ValueError("Target product missing from full catalog: " + asin)
                    actual = re.sub(r"\s+", "", json.loads(row[0])["Title"]).casefold()
                    if expected not in actual:
                        raise ValueError("Target title does not match its mapped product: " + asin)
            write_json(staging / "targets.json", target_entries)
        # Only human-goal products (roughly 10k), rather than the full catalog,
        # are passed to the original task generator.
        if goal_builder is None:
            sys.dont_write_bytecode = True
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            os.environ["WEBSHOP_DATA_MODE"] = "full"
            sys.path.insert(0, str(Path(webshop_source).resolve()))
            from web_agent_site.engine.goal import get_goals
            goal_builder = get_goals
        human_products = [json.loads(row[0]) for asin in human
                          for row in conn.execute("SELECT data FROM products WHERE asin=?", (asin,))]
        human_prices = {p["asin"]: conn.execute("SELECT price FROM products WHERE asin=?", (p["asin"],)).fetchone()[0]
                        for p in human_products}
        random.seed(seed)
        goals = goal_builder(human_products, human_prices)
        random.Random(233).shuffle(goals)
        if not goals:
            raise ValueError("Full catalog has no usable human shopping goals")
        required_asins = {g["asin"] for g in goals}
        if target_entries is not None:
            required_asins.update(a for pair in target_entries for a in pair["target_asins"])
        if required_asins.intersection(empty_asins):
            raise ValueError("Required goal/target product has empty search contents")
        write_json(staging / "goals.json", goals)
        manifest = {"version": 2, "scope": "full_catalog", "selection": "all_valid_unique_source_products",
                    "products": count, "source_records_seen": seen, "skipped_invalid_or_duplicate": skipped,
                    "goal_count": len(goals), "seed": seed, "goals_sha256": fingerprint(goals),
                    "normalization_stats": normalization_stats,
                    "source_files": {name: {"path": str(path), "bytes": path.stat().st_size,
                                             "mtime_ns": path.stat().st_mtime_ns} for name, path in source_paths.items()}}
        if target_entries is not None:
            manifest["targets_sha256"] = fingerprint(target_entries)
            manifest["targets_verified"] = True
        manifest["bundle_id"] = fingerprint(manifest)
        index = {"bundle_id": manifest["bundle_id"], "backend": "sqlite_fts5" if use_fts else "lucene",
                 "documents": documents, "empty_product_asins": empty_asins,
                 "ranking": "sqlite_fts5_bm25_or_terms" if use_fts else "lucene_default",
                 "path": "catalog.sqlite" if use_fts else str(Path(index_path).resolve()) if index_path else "index"}
        write_json(staging / "data_manifest.json", manifest)
        if not use_fts and index_path is None:
            conn.close()
            _build_lucene_index(staging, documents, progress)
            index["java_heap_mb"] = 768
            index["index_verified"] = True
        else:
            conn.close()
        write_json(staging / "index_manifest.json", index)
        staging.rename(destination)
        return manifest
    except Exception:
        if docs_stream is not None:
            docs_stream.close()
        conn.close()
        raise


def _build_lucene_index(root, expected, progress=None):
    """Use the existing Anserini Java indexer and verify in a separate process.

    The subprocess is important on Windows: all Lucene mappings are closed
    before the current-project staging directory is atomically published.
    """
    from src.environment.local_webshop import configure_java
    configure_java()
    env = os.environ.copy()
    env["JAVA_TOOL_OPTIONS"] = "-Xms64m -Xmx768m"
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "not-used")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    spec = importlib.util.find_spec("pyserini")
    jars = list((Path(spec.origin).parent / "resources" / "jars").glob("*fatjar.jar")) if spec else []
    if len(jars) != 1:
        raise RuntimeError("Cannot uniquely locate the installed Anserini Java indexer")
    java = str(Path(os.environ["JAVA_HOME"]) / "bin" / "java.exe") if os.name == "nt" else "java"
    cmd = [java, "-cp", str(jars[0]), "io.anserini.index.IndexCollection",
           "-collection", "JsonCollection", "-input", str(root / "search_documents"),
           "-index", str(root / "index"), "-generator", "DefaultLuceneDocumentGenerator",
           "-threads", "1", "-memoryBuffer", "128", "-storePositions", "-storeDocvectors", "-storeRaw"]
    with (root / "index_build.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                time.sleep(10)
                if progress is not None:
                    progress({"phase": "lucene_index", "expected_documents": expected})
            if process.returncode:
                raise RuntimeError("Lucene index build failed; see index_build.log")
        except BaseException:
            process.terminate()
            process.wait()
            raise
    verify = subprocess.run([sys.executable, "-c",
        "import sys; from pyserini.search.lucene import LuceneSearcher; "
        "s=LuceneSearcher(sys.argv[1]); print(s.num_docs); s.close()", str(root / "index")],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if verify.returncode or not verify.stdout.strip():
        raise RuntimeError("Cannot verify completed Lucene index: " + verify.stderr[-1000:])
    actual = int(verify.stdout.strip().splitlines()[-1])
    if actual != expected:
        raise ValueError(f"Lucene document mismatch: {actual} != {expected}")
