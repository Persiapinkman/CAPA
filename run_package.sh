# 添加元数据
python3 util/package_skill.py .cursor/skills/video-frame-extract \
    --version 1.0.0 \
    --exclude "*.tmp,test_data" \
    --metadata author="xiaokun1@sensetime.com" \
    --output dist/

python3 util/package_skill.py .cursor/skills/qwen-vlm-open-set-delection \
    --version 1.0.0 \
    --exclude "*.tmp,test_data" \
    --metadata author="sushuting@sensetime.com" \
    --output dist/