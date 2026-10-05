"""Current-project server adapter for disk catalogs; sibling source is read-only."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from src.environment.disk_catalog import DiskCatalog, build_catalog, read_json


def _mapping(value):
    if value is None:
        return {}
    if isinstance(value, (str, Path)):
        return read_json(value)
    if not isinstance(value, dict):
        raise TypeError("Registry and image overrides must be mappings or JSON file paths")
    return value


def _goal(value, catalog):
    if isinstance(value, int):
        if value < 0 or value >= len(catalog.goals):
            raise ValueError("Goal index outside catalog")
        value = catalog.goals[value]
    elif isinstance(value, dict) and set(value) == {"goal_index"}:
        return _goal(value["goal_index"], catalog)
    if not isinstance(value, dict):
        raise ValueError("A registered session needs a real goal object or goal index")
    required = {"asin", "category", "query", "name", "product_category", "instruction_text",
                "attributes", "price_upper", "goal_options"}
    if not required.issubset(value):
        raise ValueError("Registered goal missing fields: " + str(sorted(required - set(value))))
    if value["asin"] not in catalog.products:
        raise ValueError("Registered goal product is absent from the catalog")
    return json.loads(json.dumps(value))


class ImageProductMap:
    """Apply only a main-image overlay to a freshly decoded product."""
    def __init__(self, catalog, image_overrides, asset_root, transform=None):
        self.catalog, self.image_overrides, self.asset_root = catalog, image_overrides, asset_root
        self.transform = transform

    def __len__(self):
        return self.catalog.count

    def __iter__(self):
        return iter(self.catalog.products)

    def __contains__(self, key):
        return key in self.catalog.products

    def __getitem__(self, asin):
        product = self.catalog.product(asin)
        override = _mapping(self.image_overrides).get(asin)
        if override:
            if isinstance(override, dict):
                override = override.get("url") or override.get("SourceURL") or override.get("path")
            if not isinstance(override, str):
                raise ValueError("Image override must contain url, SourceURL, or path")
            if not override.startswith(("http://", "https://", "/minja-assets/")):
                if self.asset_root is None:
                    raise ValueError("Local image overlays require asset_root")
                path = Path(override)
                path = (self.asset_root / path).resolve() if not path.is_absolute() else path.resolve()
                if not path.is_relative_to(self.asset_root) or not path.is_file():
                    raise ValueError("Image overlay must exist within asset_root")
                override = "/minja-assets/" + path.relative_to(self.asset_root).as_posix()
            product["MainImage"] = override
            product["images"] = [override] + list(product.get("images", []))[1:]
        return self.transform(product) if self.transform else product

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def make_site(catalog_root, webshop_source, port=3000, java_mb=512, product_transform=None,
              task_registry=None, asset_root=None, image_overrides=None):
    """Return (source site module, metadata), with all product access on disk.

    task_registry: {session_id: full_goal_object | integer_goal_index |
                    {"goal_index": index}}, or an atomically replaced JSON file.
    image_overrides: {asin: source_url | local_path | {"url": ...} | {"path": ...}},
                     or an atomically replaced JSON file. Clearing it restores
                     the original catalog images without touching the database.
    """
    source = Path(webshop_source).resolve()
    if not (source / "web_agent_site" / "app.py").is_file():
        raise FileNotFoundError("WebShop source app is absent: " + str(source))
    existing = sys.modules.get("web_agent_site")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Another WebShop source is already imported; use an isolated server process")
    catalog = DiskCatalog.open(catalog_root)
    try:
        # Prevent imports from creating bytecode inside the read-only source.
        sys.dont_write_bytecode = True
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["WEBSHOP_DATA_MODE"] = "full"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["JAVA_TOOL_OPTIONS"] = f"-Xms64m -Xmx{int(java_mb)}m"
        from src.environment.local_webshop import configure_java
        configure_java()
        sys.path.insert(0, str(source))
        site = importlib.import_module("web_agent_site.app")
        if getattr(site.app, "_minja_disk_catalog", None) is not None:
            raise RuntimeError("This server app already has a disk catalog; use a new process")
        if catalog.backend == "lucene":
            from pyserini.search.lucene import LuceneSearcher
            index = Path(catalog.index_manifest["path"])
            if not index.is_absolute():
                index = catalog.root / index
            if not index.is_dir():
                raise FileNotFoundError(index)
            searcher = LuceneSearcher(str(index))
            if searcher.num_docs != catalog.index_manifest["documents"]:
                raise ValueError("Lucene index document count mismatch")
            if searcher.num_docs + len(catalog.index_manifest.get("empty_product_asins", [])) != catalog.count:
                raise ValueError("Lucene/catalog coverage mismatch")
        else:
            # Source index() only checks that the search engine is initialized.
            searcher = object()
        assets = Path(asset_root).resolve() if asset_root is not None else None
        if assets is not None and not assets.is_dir():
            raise FileNotFoundError(assets)
        products = ImageProductMap(catalog, image_overrides, assets, product_transform)
        site.product_item_dict, site.product_prices = products, catalog.prices
        site.all_products, site.attribute_to_asins = catalog.sequence, catalog.attributes
        site.search_engine = searcher
        site.goals, site.weights = catalog.goals, [g.get("weight", 1) for g in catalog.goals]
        site.SHOW_ATTRS_TAB = True
        site.user_sessions.clear()

        def forbid_full_load(*args, **kwargs):
            raise RuntimeError("Disk mode forbids engine.load_products; the server must stay disk-backed")
        site.load_products = forbid_full_load

        def search(keywords, ignored_searcher, ignored_sequence, ignored_products, ignored_attributes=None):
            results = catalog.search(keywords, searcher, k=50)
            # Ordinary search is at most 50 rows. Special queries return a lazy
            # sequence whose source pagination slices at most ten rows.
            if isinstance(results, list):
                return [products[p["asin"]] for p in results]
            return _OverlaySequence(results, products)
        site.get_top_n_product_from_keywords = search
        from flask import abort, jsonify, request, send_from_directory

        @site.app.before_request
        def initialize_real_goal():
            sid = (request.view_args or {}).get("session_id") or request.args.get("session_id")
            if sid is None:
                return None
            registry = _mapping(task_registry)
            if sid in registry and (request.endpoint == "index" or sid not in site.user_sessions):
                site.user_sessions[sid] = {"goal": _goal(registry[sid], catalog), "done": False}
            elif request.endpoint == "index" and sid.startswith("fixed_"):
                site.user_sessions.pop(sid, None)
            # Fresh index requests restore the original episode; registry goals
            # are real reward goals, never instruction-only display overrides.
            if len(site.user_sessions) > 100:
                for key in list(site.user_sessions)[:50]:
                    if key != sid:
                        site.user_sessions.pop(key, None)
            return None

        @site.app.route("/minja-disk/health")
        def disk_health():
            return jsonify({"version": 1, "port": port, "webshop_source": str(source),
                            **identity, **catalog.diagnostics()})

        @site.app.route("/minja-assets/<path:filename>")
        def asset(filename):
            if assets is None:
                abort(404)
            requested = (assets / filename).resolve()
            if not requested.is_relative_to(assets) or not requested.is_file():
                abort(404)
            return send_from_directory(str(assets), requested.relative_to(assets).as_posix())

        site.app._minja_disk_catalog = catalog
        identity = {"task_registry": str(Path(task_registry).resolve()) if isinstance(task_registry, (str, Path)) else None,
                    "asset_root": str(assets) if assets is not None else None,
                    "image_overrides": str(Path(image_overrides).resolve()) if isinstance(image_overrides, (str, Path)) else None}
        metadata = {"version": 1, "port": port, "webshop_source": str(source), **identity, **catalog.diagnostics()}
        return site, metadata
    except Exception:
        catalog.close()
        raise


class _OverlaySequence:
    def __init__(self, sequence, products):
        self.sequence, self.products = sequence, products

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        found = self.sequence[index]
        if isinstance(index, slice):
            return [self.products[p["asin"]] for p in found]
        return self.products[found["asin"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--catalog", required=True)
    serve.add_argument("--webshop-source", required=True)
    serve.add_argument("--port", type=int, default=3000)
    serve.add_argument("--java-mb", type=int, default=512)
    serve.add_argument("--task-registry")
    serve.add_argument("--asset-root")
    serve.add_argument("--image-overrides")
    build = commands.add_parser("build")
    build.add_argument("--raw-items", required=True)
    build.add_argument("--raw-attributes", required=True)
    build.add_argument("--human-goals", required=True)
    build.add_argument("--destination", required=True)
    build.add_argument("--webshop-source", required=True)
    build.add_argument("--index-path")
    build.add_argument("--search-backend", choices=["lucene", "sqlite_fts5"], default="lucene")
    build.add_argument("--targets")
    args = parser.parse_args()
    if args.stage == "build":
        result = build_catalog(args.raw_items, args.raw_attributes, args.human_goals,
                               args.destination, args.webshop_source, index_path=args.index_path,
                               search_backend=args.search_backend, targets=args.targets,
                               progress=lambda info: print(json.dumps(info), flush=True))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        site, metadata = make_site(args.catalog, args.webshop_source, args.port, args.java_mb,
                                  task_registry=args.task_registry, asset_root=args.asset_root,
                                  image_overrides=args.image_overrides)
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
        site.app.run(host="127.0.0.1", port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
