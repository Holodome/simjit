# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import base64
import binascii
import dataclasses
import platform
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .benchmark_model import BenchmarkErrorResult
from .demo_data import demo_context
from .limits import public_limits
from .service import (
    BENCHMARK_PROVIDERS,
    CompileResult,
    INPUT_MODES,
    BenchmarkOptions,
    DEFAULT_QUERY,
    benchmark_availability,
    benchmark_availability_matrix,
    benchmark_target_choices,
    compile_query,
    default_platform_arch,
    default_benchmark_target,
    normalize_arch,
    platform_choices,
    run_benchmark,
)
from .samples import DEFAULT_SAMPLE_ID, SAMPLE_GROUPS, SAMPLES_BY_ID


ROOT = Path(__file__).resolve().parent
STATIC_VERSION = "system-css-400"

app = FastAPI(title="Simjit Explorer")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


class RequestGuardError(ValueError):
    pass


@app.middleware("http")
async def add_security_guards(request: Request, call_next):
    if request.method == "POST" and not _is_same_origin_post(request):
        return PlainTextResponse("cross-origin POST rejected", status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = _cache_control_for_request(request)
    return response


def _cache_control_for_request(request: Request) -> str:
    return _cache_control_for_path(_canonical_request_path(request))


def _canonical_request_path(request: Request) -> str:
    root_path = str(request.scope.get("root_path") or "")
    if root_path.rstrip("/").endswith("/static"):
        return "/static"
    raw_paths = [str(request.scope.get("path") or ""), request.url.path]
    for raw_path in raw_paths:
        if not raw_path:
            continue
        candidates = [raw_path]
        if root_path and raw_path.startswith(root_path):
            candidates.append(raw_path[len(root_path) :] or "/")
        for candidate in candidates:
            if candidate.startswith("/static") or candidate in {"/demo", "/favicon.svg"}:
                return candidate
    return raw_paths[0] or "/"


def _cache_control_for_path(path: str) -> str:
    if path.startswith("/static"):
        return "public, max-age=3600"
    if path == "/favicon.svg":
        return "public, max-age=3600"
    if path == "/demo":
        return "public, max-age=300"
    return "no-store"


def _host_family() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm"
    if machine in {"x86_64", "amd64"}:
        return "x86"
    return "unknown"


def section_language(title: str, arch: str = "native") -> str:
    normalized_arch = normalize_arch(arch)
    if title == "AsmJit":
        family = _host_family() if normalized_arch == "native" else normalized_arch
        return "language-armasm" if family == "arm" else "language-x86asm"
    return {
        "Serialized HIR": "language-lisp",
        "HIR": "language-simjit-hir",
        "Vectorizer": "language-simjit-vectorizer",
        "MIR": "language-simjit-mir",
        "C++": "language-cpp",
        "LLVM IR": "language-llvm",
    }.get(title, "nohighlight")


templates.env.globals["section_language"] = section_language
templates.env.globals["static_version"] = STATIC_VERSION


@app.get("/favicon.svg")
async def favicon_svg():
    return FileResponse(ROOT / "static" / "simjit-favicon.svg", media_type="image/svg+xml")


def _int_param(params: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(params.get(name, default)))
    except (TypeError, ValueError):
        return default


def _float_param(params: Mapping[str, str], name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        return min(maximum, max(minimum, float(params.get(name, default))))
    except (TypeError, ValueError):
        return default


def _benchmark_options_from_params(params: Mapping[str, str]) -> BenchmarkOptions:
    return BenchmarkOptions(
        rows=_int_param(params, "rows", 100_000, 1),
        warmups=_int_param(params, "warmups", 3, 0),
        runs=_int_param(params, "runs", 10, 1),
        null_density=_float_param(params, "null_density", 0.1, 0.0, 1.0),
        output=str(params.get("output", "pyarrow")),
        arch=str(params.get("arch", "native")),
        benchmark_target=str(params.get("benchmark_target", default_benchmark_target())),
        provider=str(params.get("provider", "google")),
    )


def _benchmark_options_from_form(params: Mapping[str, str]) -> BenchmarkOptions:
    caps = public_limits()
    return BenchmarkOptions(
        rows=_strict_int_param(params, "rows", 100_000, 1),
        warmups=_strict_int_param(params, "warmups", 3, 0),
        runs=_strict_int_param(params, "runs", 10, 1),
        null_density=_strict_float_param(params, "null_density", 0.1, 0.0, caps.max_null_density),
        output=str(params.get("output", "pyarrow")),
        arch=str(params.get("arch", "native")),
        benchmark_target=str(params.get("benchmark_target", default_benchmark_target())),
        provider=str(params.get("provider", "google")),
    )


def _strict_int_param(params: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = params.get(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RequestGuardError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RequestGuardError(f"{name} must be at least {minimum}")
    return value


def _strict_float_param(params: Mapping[str, str], name: str, default: float, minimum: float, maximum: float) -> float:
    raw = params.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RequestGuardError(f"{name} must be a number") from exc
    if value < minimum or value > maximum:
        raise RequestGuardError(f"{name} must be between {minimum:g} and {maximum:g}")
    return value


def _sample_option_query(options: BenchmarkOptions) -> str:
    default_options = _default_benchmark_options()
    params = {}
    if options.benchmark_target != default_options.benchmark_target:
        params["benchmark_target"] = options.benchmark_target
    if options.provider != default_options.provider:
        params["provider"] = options.provider
    if options.rows != default_options.rows:
        params["rows"] = options.rows
    if options.warmups != default_options.warmups:
        params["warmups"] = options.warmups
    if options.runs != default_options.runs:
        params["runs"] = options.runs
    if options.null_density != default_options.null_density:
        params["null_density"] = options.null_density
    if options.output != default_options.output:
        params["output"] = options.output
    try:
        normalized_arch = normalize_arch(options.arch)
        default_arch = default_platform_arch(options.benchmark_target)
        if normalized_arch not in {"native", default_arch}:
            params["arch"] = options.arch
    except ValueError:
        params["arch"] = options.arch
    return urlencode(params)


def _default_benchmark_options() -> BenchmarkOptions:
    return BenchmarkOptions(benchmark_target=default_benchmark_target())


def _url_default_values() -> dict[str, str]:
    options = _default_benchmark_options()
    return {
        "benchmark_target": options.benchmark_target,
        "provider": options.provider,
        "rows": str(options.rows),
        "warmups": str(options.warmups),
        "runs": str(options.runs),
        "null_density": f"{options.null_density:g}",
        "output": options.output,
        "input_mode": "expression_sql",
    }


def _template_response(request: Request, name: str, context: dict):
    try:
        return templates.TemplateResponse(request=request, name=name, context=context)
    except TypeError:
        return templates.TemplateResponse(name, context)


def _render(
    request: Request,
    *,
    query: str = DEFAULT_QUERY,
    result=None,
    benchmark=None,
    options: BenchmarkOptions | None = None,
    arch: str = "native",
    benchmark_target: str | None = None,
    input_mode: str = "expression_sql",
    selected_sample_id: str | None = None,
):
    try:
        selected_arch = normalize_arch(arch)
    except ValueError:
        selected_arch = "native"
    selected_target = benchmark_target or default_benchmark_target()
    selected_platform = selected_arch if selected_arch != "native" else default_platform_arch(selected_target)
    render_options = options or _default_benchmark_options()
    availability = benchmark_availability(input_mode, selected_platform, selected_target)
    app_base = request.scope.get("root_path") or ""
    return _template_response(
        request,
        "index.html",
        {
            "request": request,
            "app_base": app_base,
            "query": query,
            "sample_groups": SAMPLE_GROUPS,
            "selected_sample_id": selected_sample_id,
            "result": result,
            "benchmark": benchmark,
            "options": render_options,
            "arch": selected_arch,
            "platform_arch": selected_platform,
            "platforms": platform_choices(),
            "benchmark_targets": benchmark_target_choices(),
            "benchmark_target": selected_target,
            "benchmark_providers": BENCHMARK_PROVIDERS,
            "benchmark_availability": availability,
            "benchmark_availability_matrix": benchmark_availability_matrix(),
            "sample_option_query": _sample_option_query(render_options),
            "url_defaults": _url_default_values(),
            "input_modes": INPUT_MODES,
            "input_mode": input_mode,
            "benchmark_runnable": availability.runnable,
        },
    )


def _render_results(
    request: Request,
    *,
    result=None,
    benchmark=None,
    options: BenchmarkOptions | None = None,
):
    return _template_response(
        request,
        "_results.html",
        {
            "request": request,
            "result": result,
            "benchmark": benchmark,
            "options": options or BenchmarkOptions(),
            "benchmark_providers": BENCHMARK_PROVIDERS,
        },
    )


def _wants_results_fragment(request: Request) -> bool:
    return request.headers.get("X-Simjit-Explorer-Fragment") == "results"


def _error_result(query: str, message: str, *, arch: str = "native", input_mode: str = "expression_sql") -> CompileResult:
    return CompileResult(False, query, {}, [], error=message, arch=arch, input_mode=input_mode)


def _is_same_origin_post(request: Request) -> bool:
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    candidate = origin or referer
    if not candidate:
        return True
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        return False
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("host") or request.url.netloc
    return f"{parsed.scheme}://{parsed.netloc}" == f"{scheme}://{host}"


async def _limited_form(request: Request):
    caps = public_limits()
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise RequestGuardError("invalid content length") from exc
        if length > caps.max_form_bytes:
            raise RequestGuardError(f"form body is too large ({length} bytes > {caps.max_form_bytes} bytes)")
    form = await request.form()
    if len(form) > caps.max_form_fields:
        raise RequestGuardError(f"too many form fields ({len(form)} > {caps.max_form_fields})")
    for name, value in form.multi_items():
        if not isinstance(value, str):
            raise RequestGuardError("file uploads are not supported")
        size = len(value.encode("utf-8"))
        maximum = caps.max_query_bytes if name == "query" else caps.max_form_value_bytes
        if size > maximum:
            raise RequestGuardError(f"{name} is too large ({size} bytes > {maximum} bytes)")
    return form


def decode_shared_query(encoded: str) -> str:
    caps = public_limits()
    value = encoded.strip().replace(" ", "+")
    if len(value.encode("utf-8")) > caps.max_share_query_bytes:
        raise ValueError(
            f"base64 query parameter is too large ({len(value.encode('utf-8'))} bytes > "
            f"{caps.max_share_query_bytes} bytes)"
        )
    padding = "=" * ((4 - len(value) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(value + padding).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("invalid base64 query parameter") from exc
    size = len(decoded.encode("utf-8"))
    if size > caps.max_query_bytes:
        raise ValueError(f"shared query is too large ({size} bytes > {caps.max_query_bytes} bytes)")
    return decoded


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    options = _benchmark_options_from_params(request.query_params)
    shared_query = request.query_params.get("query")
    if shared_query is not None:
        input_mode = request.query_params.get("input_mode") or "expression_sql"
        try:
            query = decode_shared_query(shared_query)
            result = compile_query(query, arch=options.arch, input_mode=input_mode)
        except ValueError as exc:
            query = ""
            result = compile_query(query, arch=options.arch, input_mode=input_mode)
            result = dataclasses.replace(result, error=str(exc))
        return _render(
            request,
            query=query,
            result=result,
            options=options,
            arch=result.arch,
            benchmark_target=options.benchmark_target,
            input_mode=result.input_mode,
        )

    requested = request.query_params.get("example") or DEFAULT_SAMPLE_ID
    sample = SAMPLES_BY_ID.get(requested, SAMPLES_BY_ID[DEFAULT_SAMPLE_ID])
    result = compile_query(sample.query, arch=options.arch, input_mode=sample.input_mode)
    return _render(
        request,
        query=sample.query,
        result=result,
        options=options,
        arch=result.arch,
        benchmark_target=options.benchmark_target,
        input_mode=sample.input_mode,
        selected_sample_id=sample.id,
    )


@app.get("/demo", response_class=HTMLResponse)
async def demo(request: Request):
    app_base = request.scope.get("root_path") or ""
    demo_home_href = request.headers.get("x-simjit-demo-home") or (f"{app_base}/demo" if app_base else "/demo")
    return _template_response(
        request,
        "demo.html",
        {
            "request": request,
            "app_base": app_base,
            "demo_home_href": demo_home_href,
            **demo_context(),
        },
    )


@app.head("/demo")
async def demo_head():
    return HTMLResponse("")


@app.get("/examples/{name}")
async def example(name: str):
    if name not in SAMPLES_BY_ID:
        return RedirectResponse("../", status_code=303)
    return RedirectResponse(f"../?example={name}", status_code=303)


@app.post("/compile", response_class=HTMLResponse)
async def compile_form(request: Request):
    try:
        form = await _limited_form(request)
    except RequestGuardError as exc:
        result = _error_result("", str(exc))
        if _wants_results_fragment(request):
            return _render_results(request, result=result)
        return _render(request, result=result)
    query = str(form.get("query", DEFAULT_QUERY))
    arch = str(form.get("arch", "native"))
    input_mode = str(form.get("input_mode", "expression_sql"))
    benchmark_target = str(form.get("benchmark_target", default_benchmark_target()))
    result = compile_query(query, arch=arch, input_mode=input_mode)
    if _wants_results_fragment(request):
        return _render_results(request, result=result)
    return _render(
        request,
        query=query,
        result=result,
        arch=result.arch,
        benchmark_target=benchmark_target,
        input_mode=result.input_mode,
    )


@app.post("/benchmark", response_class=HTMLResponse)
async def benchmark_form(request: Request):
    try:
        form = await _limited_form(request)
    except RequestGuardError as exc:
        benchmark = BenchmarkErrorResult(str(exc))
        if _wants_results_fragment(request):
            return _render_results(request, benchmark=benchmark)
        return _render(request, benchmark=benchmark)
    query = str(form.get("query", DEFAULT_QUERY))
    input_mode = str(form.get("input_mode", "expression_sql"))
    try:
        options = _benchmark_options_from_form(form)
    except RequestGuardError as exc:
        benchmark = BenchmarkErrorResult(str(exc))
        if _wants_results_fragment(request):
            return _render_results(request, benchmark=benchmark)
        return _render(request, query=query, benchmark=benchmark, input_mode=input_mode)
    result = compile_query(query, arch=options.arch, input_mode=input_mode)
    benchmark = (
        run_benchmark(query, options)
        if result.ok and result.input_mode == "expression_sql"
        else BenchmarkErrorResult("Benchmarks require Expression SQL input mode")
        if result.ok
        else None
    )
    if _wants_results_fragment(request):
        return _render_results(
            request,
            result=result,
            benchmark=benchmark,
            options=options,
        )
    return _render(
        request,
        query=query,
        result=result,
        benchmark=benchmark,
        options=options,
        arch=result.arch,
        benchmark_target=options.benchmark_target,
        input_mode=result.input_mode,
    )
