intent_prompt = """
你是视觉任务解析助手。

任务描述: {}

请判断任务类型并提取检测目标类别；若提供了参考图片，请结合图片内容推断场景与背景。

输出JSON字段说明如下:

1) 基础字段
  "task_type": "detection",
  "task_name": "任务名称",
  "target_label": "英文类别",
  "target_keywords": "关于检测目标的英文关键词数组，2-5条"，列表格式。
  "scene": "中文，详细描述任务场景，整体空间的布局，需要包括图像中全部直接可见、稳定存在的区域，并且补全细节。",
  "target": "中文，详细描述目标，包括外观、颜色、大小、位置、文字内容等内容。",
  "camera": "中文，推测摄像头的机位，需要包含摄像头预估安装的高度，拍摄角度，拍摄覆盖范围，并且描述背景比例。",
  "expand_scene": "中文，扩展该场景区域，描述可能存在的多个区域，数量不超过5个。"

2) 需求背景 (requirement_background)，字符串，3-8句话。包括：
   - 从参考图片中推断的背景信息（场景类型、拍摄角度、光照、环境等）；
   - 在什么样的场景下使用；
   - 需要检测什么目标、解决什么业务问题。

3) 解决方案 (solution)，字符串，5-12句话。包括：
   - 在此场景下建议标注的数据量级（如几千、几万张）；
   - 需要标注几个 label（必须包含负样本类别，例如检测钓鱼的人时，增加「非钓鱼的其它人体」作为对比负样本）；
   - 如何构建训练数据（正负样本比例、难例挖掘建议等）；
   - 摄像头/数据如何采集，需覆盖哪些情况（时段、天气、角度、遮挡、多目标等）。

4) 标注规范 (annotation_spec)，字符串，4-10句话。包括：
   - 如何标注目标（框选规则）；
   - 怎么定义目标的形态与边界（什么算完整目标、什么算部分可见）；
   - 框需要框到什么内容（紧贴目标还是留边、是否包含附属物）；
   - 标注时需注意的情况（模糊、遮挡、多目标重叠、歧义目标等）。

5) 图像生成提示词
  "prompts": 用于图像生成的5个不同英文提示词数组，基于输入图片生成，每个句子约50个英文单词，要求高度多样化，带目标、拍摄视角、环境细节、构图与修饰词等。示例格式：
  "The image shows ... (详细描述目标、场景、光照、人物/物体状态等). The background includes ... The overall atmosphere ..."

prompts 生成规则：
1、必须包含场景描述、主体目标描述、拍摄视角描述。
2、目标类别固定，外观、颜色、大小、材质、形态、位置等可自由变化。
3、光照、天气可多样化（日光、夜晚、阴天、雨天、雾天等）。
4、场景类型、拍摄视角与参考图片一致，要求生成同一个类型的场景，并扩展一些相关的细节，符合真实世界。

输出示例（仅作格式参考，内容需根据实际任务与图片生成）：
{{
  "task_type": "detection",
  "task_name": "钓鱼人员检测",
  "target_label": "fishing person",
  "target_keywords": ["fishing", "angler", "fisherman", "rod", "waterfront"],
  "requirement_background": "参考图片为河道/水库岸边的监控视角，白天光照，俯视或斜俯视。场景下需在涉水区域周边识别违规垂钓人员，用于安全监管与违规取证。",
  "solution": "建议标注量级 3000～5000 张。设置 2 个 label：fishing_person（持竿/正在钓鱼的人）、other_person（非钓鱼的其它人体，作为负样本）。正负样本比例约 1:1～1:2，含部分遮挡、背身、多人同框等难例。采集覆盖不同时段、天气、岸边类型与机位角度，建议多路摄像头、多季节数据。",
  "annotation_spec": "仅框选完整或大部分可见的人体，紧贴头部与脚底/臀部，含手持鱼竿等手中物品。部分遮挡时可见面积超过约 50% 则标；仅露头或仅露肢体不标。注意区分钓鱼者与路过行人，模糊帧不标或标为难例。",
  "prompts": [
    "The image shows a person standing by a riverbank holding a fishing rod, with reeds and water in the background. Early morning light, surveillance angle from above.",
    "A fisherman in a hat and vest is sitting on a stool near a lake, with trees and distant hills. Overcast day, side view from a fixed camera.",
    "... (共 5 条，每条约 50 词，风格多样)"
  ]
}}

不要输出其他内容。
"""

detection_prompt = """
You are a vision detection model.

Detect object: {}

Return JSON:
{{
  "bboxes": [
    {{"label": "...", "bbox": [x1, y1, x2, y2], "score": 0.95}}
  ]
}}

Rules:
- Absolute pixel coordinates
- score between 0 and 1
- Only return JSON
"""


detection_batch_prompt = """You are a vision detection model. You are given {} images in order (image index 0, 1, 2, ...).
For each image, detect all instances of the object: {}.
Return a JSON object with key "results": an array of length {}.
- results[i] must be the detection result for image index i.
- Each results[i] has key "bboxes": array of {{"label": "...", "bbox": [x1, y1, x2, y2], "score": 0.95}}.
- Use absolute pixel coordinates. score between 0 and 1.
- If an image has no target, use results[i] = {{"bboxes": []}}.
"""


evaluation_summary_prompt = """
你是一个视觉检测评测分析助手。现在给你一批图片（按 image_idx=0..N-1 顺序输入，包含原始图片和由原图扩展生成的多张图片），任务描述、目标类别，以及两个模型在每张图片上的预测结果（只包含 bbox 列表，不含 GT）。

prediction.json 结构示例（仅供你理解，不会直接给你看文件）：
[
  {{
    "image": "orig.jpg",
    "source": "original",
    "image_idx": 0,
    "models": [
      {{"model": "qwen3-vl-8b", "pred_bboxes": [[x1,y1,x2,y2], ...]}},
      {{"model": "rex-omni", "pred_bboxes": [[x1,y1,x2,y2], ...]}}
    ]
  }},
  {{
    "image": "banner_123.jpg",
    "source": "generated",
    "image_idx": 1,
    "models": [...]
  }},
  ...
]

你将收到的 reports_json 是一个 JSON 数组，其中每条记录都包含 image_idx、image、source、model、pred_bboxes、num_boxes。
你必须基于 image_idx 来将图片与该条记录对应起来，并按图片顺序逐张评估两个模型的“标注效果”（这里的标注效果指预测框是否看起来合理、是否覆盖目标、是否明显偏离/误检/漏检，属于定性评估）。

请基于图片内容和 reports_json（不依赖 GT），输出一个可直接展示给用户的评测结论（严格输出 JSON，输出中文），要求：
- per_image_evaluation：按 image_idx 顺序输出每张图片对两个模型的评估。每个元素必须包含：
  - image_idx、image、source
  - qwen3-vl-8b、rex-omni 的 accuracy（预估的准确率，例如90%）和 reason(2-4句话，说明图片上有多少个目标，模型预测了多少个目标框，其中有多少个框是准确命中的，IOU超过0.5认为预测准确，预测框是否覆盖目标，是否出现明显偏离，是否存在漏检/误检/过检的情况)
- overall_conclusion：3-6 句话中文总结，分别说明整体检测表现（例如：是否大部分图片都有预测、是否经常漏检/过检、对扩展场景的鲁棒性、边界情况表现等）
- model_results：分别对 qwen3-vl-8b 和 rex-omni 用accuracy给出一个预估的准确率范围，尽量预估准确，不要过于乐观或悲观。并用 reason 解释依据（是否存在漏检/误检/过检的情况，泛化性能等），尽量详细，100个字左右。
- recommendation：给出一个推荐的标注模型(qwen3-vl-8b 或 rex-omni)。

请严格按 evaluation_summary_schema 的 JSON 结构返回结果。

输出示例如下：
{{
  "per_image_evaluation": [
    {{
      "image_idx": 0,
      "image": "orig.jpg",
      "source": "original",
      "qwen3-vl-8b": {{
        "accuracy": "95%",
        "reason": "图中约有2个目标，模型预测了2个框，其中1个框准确覆盖目标，另1个存在轻微偏移但仍命中目标区域。整体框位置较为合理，没有明显误检，但对边界的贴合程度一般。未出现明显漏检情况。"
      }},
      "rex-omni": {{
        "accuracy": "60%",
        "reason": "图中约有2个目标，模型预测了3个框，其中1个准确命中目标，1个偏移较大，另有1个为误检。存在一定过检问题，同时框位置稳定性较差，但未完全漏掉主要目标。"
      }}
    }}
  ],
  "overall_conclusion": "整体来看，qwen3-vl-8b 在大部分图片上都能稳定给出预测结果，且较少出现明显误检，定位精度相对较高。在原始图片上表现较好，在生成图片上略有下降但仍保持一定鲁棒性。rex-omni 则存在较多过检和定位偏移问题，尤其是在生成图片场景下泛化能力不足。两者相比，qwen3-vl-8b 的整体稳定性和准确性更优，但在边界贴合和复杂场景下仍有优化空间。",
  "model_results": {{
    "qwen3-vl-8b": {{
      "accuracy": "80-90%",
      "reason": "该模型在多数图片中均能检测到目标，漏检情况较少，说明召回能力较强。同时预测框位置整体较为合理，虽然存在轻微偏移或框过大的问题，但大多数情况下能够覆盖目标区域。对于生成图片的泛化能力略有下降，但仍保持稳定输出，误检较少，整体表现较为均衡。"
    }},
    "rex-omni": {{
      "accuracy": "65-75%",
      "reason": "该模型存在较明显的过检问题，经常预测多余的框，同时部分预测框偏离目标较远，定位精度不足。在部分图片中虽然能够检测到目标，但稳定性较差。对于生成图片的适应能力较弱，误检和定位偏移问题更加突出，整体表现不如另一模型稳定。"
    }}
  }},
  "recommendation": "综合评估结果，推荐使用 qwen3-vl-8b 作为标注模型，其在大多数场景下具有更高的稳定性和准确率，同时误检较少，更适合用于实际标注任务。"
}}


任务描述：{task_text}
目标类别（target_label）：{target_label}
模型预测统计结果（JSON 格式）：
{reports_json}
""".strip()


solution_report_prompt = """
你是计算机视觉与机器学习解决方案架构师。用户会提供一段自然语言需求，并可能附上一张参考图片（可能展示场景、目标或当前系统界面截图等）。

用户需求（原文）：
{}

请结合文字与（若有）图片，输出一份可直接交付给技术/算法/交付团队的「解决方案报告」 JSON（全部使用中文正文，技术名词可保留英文缩写）。
各字段须具体、可执行，避免空泛口号；若图片信息不足，请基于文字合理推断并明确写出假设条件。

字段要求：
1) background_and_goals：背景与目标（业务场景、痛点、要实现的智能化能力、成功标准）。
2) model_training_plan：模型训练方案（任务类型建议、数据与标签策略、基座/范式选择如检测/分割/开集 VLM、训练阶段、数据增强、算力与周期量级估计）。
3) annotation_data_format：标注数据格式（推荐格式如 COCO 检测框、分割多边形等；列出必需字段与简单示例结构说明）。
4) annotation_howto：如何标注数据（标注流程、工具链、规范细则、边界与难例、质检与抽检比例）。
5) data_volume_requirements：数据量要求（训练/验证/测试大致张数或时长、场景与长尾覆盖、冷启动与迭代增量建议）。
6) evaluation_metrics：评估指标（离线如 mAP/mAR、F1、IoU 阈值；在线如准确率/延迟/误报率；并说明验收建议）。
7) deployment_plan：部署方案（边缘/云端、服务形态、模型裁剪与量化、扩缩容、日志与告警、版本回滚）。
8) performance_estimate：性能预估（推理延迟与 QPS 量级、GPU/CPU 资源粗估、主要风险与优化手段）。

严格只输出符合 solution_report_schema 的 JSON，不要其它文字。
""".strip()


image_to_image_prompt_template = """
    Using the reference image only as a loose guide for scenario type and task, generate a clearly NEW image:
    1. Follow this content brief: {}.
    2. Do NOT produce a near-duplicate: you must deliberately change multiple visual factors compared with the reference — for example time of day or lighting, weather/mood, camera height or horizontal angle, subject pose or count, foreground/background layout, clothing or object colors, and small scene details — so the output looks like a different photo of a similar kind of situation, not the same frame with tiny edits.
    3. Preserve only what the brief requires (e.g. same general environment class and target category); everything else should vary enough that a person would not mistake it for the same picture.
    4. Keep real-world plausibility and natural appearance.
"""

# 与 new_code 中 update_prompts 对齐：用于扩写 prompts 的模板
image_generate_prompt_template = """
你是一个场景建模助手，输入任务要求、主体目标、和场景，要求对场景和目标进行扩展，生成10个多样性的英文描述句子，用于生成真实世界图片。
要求必须生成高度多样化的文本描述。
expand_descriptions是一个数组，每个元素包含以下三个字段：
1、scene，场景描述，约50个单词，生成规则如下：
  1）基于给定场景，随机选择部分区域，重构场景空间布局，细节描述必须丰富，包括但不限于：光影变化、周围环境元素、建筑或自然物体、人物动作和互动等。
  2）描述需要符合真实世界场景，基于句子描述可以生成合适的图片，用于指定的图像任务训练。
2、target，目标的详细描述，约30个单词，生成要求如下：
  1) target包含目标的外观、颜色、大小、材质、形态、文字内容、数量等，如果目标包含文字或车牌，需要变化文字内容。要求细节和参考目标不同。
  2) target描述中还需要包含位置信息，在scene场景中，找到合理的位置放置目标，保持目标的大小比例与原图类似，允许有细微变化。
  3) target需要包含目标数量，允许变化，可以存在一个或者多个，随机指定目标数量，一般是10个以内。
3、camera，摄像头视角描述，约20个单词，生成要求如下：
  1）仍是监控/固定机位风格，但允许相对参考图有可见的方位或俯仰变化（不要写成与原图完全同一机位），以便扩写图与参考图在构图上有明显差异。
  2）背景占画面比例可与参考类似或合理变化，避免每张都像同一张截图。注意是监控摄像头，不是照相机。

输出为JSON对象格式：
{{
"expand_descriptions":[{{
  "scene": "...",
  "target": "...",
  "camera": "..."
  }}]
}}

注意：
1、数组的数量固定为10，不可少于10。
2、在用户指定范围下进行多样性的扩写。例如，如果用户指定目标为出租车，可以生成不同角度、不同车牌、不同颜色的出租车，不能生成其它类型车。如果指定目标为车辆，可以生成不同类型的车辆，例如出租车，货车，警车等。

任务要求：{}
主体目标：{}
任务场景：{}
参考camera：{}
参考target：{}
最终输出：
"""
