import shlex
import subprocess
from typing import Any, Dict, List, Optional
from result_parser import (
    parse_benchmark_add_output,
    parse_benchmark_info_output,
    parse_benchmark_list_output,
    parse_deployment_info_output,
    parse_deployment_list_output,
    print_json,
)

SERVER = "scg-adela.sensetime.com"
PROJECT_ID = 3


def _run(cmd: str, run: bool = True, capture_stdout: bool = False):
    # 每条实际执行的 adela shell；demo 子进程读 stdout 时可按此前缀回显到终端日志
    print(f"[CMD] {cmd}", flush=True)
    if run:
        kw: Dict[str, Any] = {
            "shell": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if capture_stdout:
            kw["capture_output"] = True
        result = subprocess.run(cmd, **kw)
        if result.returncode != 0:
            out = (result.stdout or "").strip()
            err = (result.stderr or "").strip()
            blocks: List[str] = []
            if out:
                blocks.append("[adela stdout]\n" + out)
            if err:
                blocks.append("[adela stderr]\n" + err)
            raw = (
                "\n\n".join(blocks)
                if blocks
                else "(adela 无 stdout/stderr，可能被 shell 包装吞掉)"
            )
            # 先展示 CLI 原始输出，再附退出码与命令，便于一眼看到 API/报错正文
            msg = (
                f"adela 命令失败（退出码 {result.returncode}）\n"
                f"原始输出：\n{raw}\n\n"
                f"命令: {cmd}"
            )
            print(f"[ADELA ERR]\n{msg}", flush=True)
            raise RuntimeError(msg)
        if capture_stdout:
            return result.stdout or ""
        return result
    return cmd


# =========================
# 部署相关
# =========================

def deployment_add(
    rawmodel_id: int,
    platform: str,
    quant_dataset_id: Optional[int] = None,
    max_batch_size: int = 8,
    run: bool = True,
):
    cmd = (
        f"adela -s {SERVER} deployment simple_add "
        f"-p {PROJECT_ID} -r {rawmodel_id} "
        f"-pl {platform} -B {max_batch_size}"
    )

    if quant_dataset_id:
        cmd += f" -q {quant_dataset_id}"

    output = _run(cmd, run=run, capture_stdout=True)
    if not run:
        return {}
    return parse_deployment_info_output(output)


def deployment_list(
    rawmodel_id: int,
    run: bool = True,
)-> List[Dict[str, str]]:
    cmd = (
        f"adela -s {SERVER} deployment list "
        f"-p {PROJECT_ID} -r {rawmodel_id}"
    )
    output = _run(cmd, run=run, capture_stdout=True)
    if not run:
        return []
    records = parse_deployment_list_output(output)
    return records
    

def deployment_info(
    deployment_id: int,
    run: bool = True,
) -> Dict[str, Any]:
    cmd = (
        f"adela -s {SERVER} deployment info "
        f"-p {PROJECT_ID} -d {deployment_id}"
    )
    output = _run(cmd, run=run, capture_stdout=True)
    if not run:
        return {}
    info = parse_deployment_info_output(output)
    return info


# =========================
# 评测相关
# =========================

def benchmark_add(
    deployment_id: int,
    config_file: str,
    run: bool = True,
):
    cmd = (
        f"adela -s {SERVER} benchmark add "
        f"-p {PROJECT_ID} -d {deployment_id} -f {shlex.quote(config_file)}"
    )
    output = _run(cmd, run=run, capture_stdout=True)
    if not run:
        return {}
    return parse_benchmark_add_output(output)


def benchmark_list(
    deployment_id: int,
    run: bool = True,
) -> List[Dict[str, str]]:
    cmd = (
        f"adela -s {SERVER} benchmark list "
        f"-p {PROJECT_ID} -d {deployment_id}"
    )
    output = _run(cmd, run=run, capture_stdout=True)
    if not run:
        return []
    records = parse_benchmark_list_output(output)
    return records


def benchmark_info(
    deployment_id: int,
    benchmark_id: int,
    run: bool = True,
) -> Dict[str, Any]:
    cmd = (
        f"adela -s {SERVER} benchmark info "
        f"-p {PROJECT_ID} -d {deployment_id} -b {benchmark_id}"
    )
    output = _run(cmd, run=run, capture_stdout=True)
    if not run:
        return {}
    data = parse_benchmark_info_output(output)
    return data

if __name__ == "__main__":
    #res = benchmark_info(166831, 122952)
    #res = benchmark_list(166831)
    #res = benchmark_info(166831, 122952)
    #res = deployment_add(51476, "cuda11.0-trt7.1-fp16-T4")
    #res = deployment_add(51476, "cuda11.0-trt7.1-int8-T4",11)
    #res = deployment_list(51339)
    #res = deployment_info(126675)
    #res = benchmark_add(166852, "/media/nvme1n1p1/sushuting/test-skills/skills/adela-cli/references/speed_eval.json")
    res = benchmark_add(131766, "/tmp/adela-cli/adela_eval_bedpkuzd.json")
    print_json(res)