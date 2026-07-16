from __future__ import annotations

import base64
import io
import os
import time

from openai import OpenAI
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def omit_model_image_payload() -> bool:
    return os.environ.get("CAPA_OMIT_MODEL_IMAGE_PAYLOAD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class VLMService:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        timeout = float(os.environ.get("DEMO_OPENAI_TIMEOUT_SECONDS", "120"))
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.last_response_metadata: dict = {}

    def generate_text(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        image_paths: list[str] | None = None,
        response_format: dict | None = None,
        model: str | None = None,
        **kwargs,
    ):
        if model is None:
            model = self.client.models.list().data[0].id

        request_messages = messages if isinstance(messages, list) and messages else [
            {"role": "user", "content": [{"type": "text", "text": str(prompt or "")}]}
        ]
        if image_paths and len(image_paths) > 0 and not omit_model_image_payload():
            user_idx = -1
            for i in range(len(request_messages) - 1, -1, -1):
                if str(request_messages[i].get("role") or "") == "user":
                    user_idx = i
                    break
            if user_idx < 0:
                request_messages.append({"role": "user", "content": []})
                user_idx = len(request_messages) - 1
            content = request_messages[user_idx].get("content")
            if not isinstance(content, list):
                content = [{"type": "text", "text": str(content or "")}] if content else []
                request_messages[user_idx]["content"] = content
            for img_path in image_paths:
                image_b64, mime = self.image_to_base64(img_path)
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{mime};base64,{image_b64}",
                    },
                }
                request_messages[user_idx]["content"].append(image_content)

        if "temperature" not in kwargs and os.environ.get("DEMO_OPENAI_TEMPERATURE"):
            kwargs["temperature"] = float(os.environ["DEMO_OPENAI_TEMPERATURE"])
        if "top_p" not in kwargs and os.environ.get("DEMO_OPENAI_TOP_P"):
            kwargs["top_p"] = float(os.environ["DEMO_OPENAI_TOP_P"])
        if "seed" not in kwargs and os.environ.get("DEMO_OPENAI_SEED"):
            kwargs["seed"] = int(os.environ["DEMO_OPENAI_SEED"])
        if "max_tokens" not in kwargs and os.environ.get("DEMO_OPENAI_MAX_TOKENS"):
            kwargs["max_tokens"] = int(os.environ["DEMO_OPENAI_MAX_TOKENS"])
        if os.environ.get("DEMO_OPENAI_DO_SAMPLE"):
            do_sample = os.environ["DEMO_OPENAI_DO_SAMPLE"].strip().lower() in {"1", "true", "yes", "on"}
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["do_sample"] = do_sample
            kwargs["extra_body"] = extra_body

        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=request_messages,
                response_format=response_format,
                **kwargs,
            )
        except Exception as exc:
            self.last_response_metadata = {
                "api_call_ms": round((time.perf_counter() - start) * 1000, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            raise
        api_call_ms = round((time.perf_counter() - start) * 1000, 3)
        usage = getattr(response, "usage", None)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        self.last_response_metadata = {
            "api_call_ms": api_call_ms,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
            "model": getattr(response, "model", None),
        }
        return response.choices[0].message.content

    def image_to_base64(self, image_path: str, threshold=1024) -> tuple[str, str]:
        with Image.open(image_path) as img:
            width, height = img.size
            save_format = (img.format or "PNG").upper()
            """
            # 检查是否需要缩小（宽或高超过阈值）
            if width > threshold or height > threshold:
                # 计算缩小一半后的尺寸
                new_width = width // 2
                new_height = height // 2
                # 高质量缩小图片
                img = img.resize((new_width, new_height), Image.LANCZOS)
            """
            # 将图片数据写入内存缓冲区
            buffer = io.BytesIO()
            img.save(buffer, format=save_format)  # 显式指定格式
            img_bytes = buffer.getvalue()

            # 转换为Base64
            base64_str = base64.b64encode(img_bytes).decode("utf-8")
            if save_format in ("JPG", "JPEG"):
                mime = "jpeg"
            else:
                mime = save_format.lower()
            return base64_str, mime
