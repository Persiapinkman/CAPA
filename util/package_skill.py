#!/usr/bin/env python3
"""
工业级 Skill 打包工具

支持版本管理、元数据、清单文件生成等功能，用于创建可分发的 .skill 文件。

用法:
    python package_skill.py <skill-path> [options]

示例:
    python package_skill.py .cursor/skills/video-frame-extract
    python package_skill.py .cursor/skills/video-frame-extract --version 1.0.0 --output dist/
    python package_skill.py .cursor/skills/video-frame-extract --version 1.0.0 --metadata author="Your Name"
"""

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# 尝试导入验证工具
try:
    sys.path.insert(0, str(Path(__file__).parent / '.cursor' / 'skills' / 'skill-creator' / 'scripts'))
    from quick_validate import validate_skill
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("⚠️  警告: 验证工具不可用，将跳过验证步骤")


# ============================================================================
# 常量定义
# ============================================================================

# 默认排除的文件和目录模式
DEFAULT_EXCLUDE_PATTERNS: Set[str] = {
    '__pycache__',
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '.git',
    '.gitignore',
    '.DS_Store',
    '*.swp',
    '*.swo',
    '*~',
    '.vscode',
    '.idea',
    '*.log',
    'logs',
    '.pytest_cache',
    '.mypy_cache',
    'dist',
    'build',
}

# 清单文件名
MANIFEST_FILE = 'MANIFEST.json'

# 支持的元数据字段
METADATA_FIELDS = {
    'name', 'version', 'description', 'author', 'license', 
    'homepage', 'repository', 'keywords', 'created_at', 'build_at'
}


# ============================================================================
# 工具函数
# ============================================================================

def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """计算文件的哈希值"""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def should_exclude_file(file_path: Path, exclude_patterns: Set[str]) -> bool:
    """判断文件是否应该被排除"""
    file_str = str(file_path)
    name = file_path.name
    
    # 检查完整路径
    for pattern in exclude_patterns:
        if pattern in file_str:
            return True
    
    # 检查文件名
    for pattern in exclude_patterns:
        if pattern.startswith('*'):
            if name.endswith(pattern[1:]):
                return True
        elif pattern == name:
            return True
    
    return False


def read_skill_metadata(skill_path: Path) -> Dict[str, str]:
    """从 SKILL.md 读取元数据"""
    metadata = {}
    skill_md = skill_path / 'SKILL.md'
    
    if not skill_md.exists():
        return metadata
    
    try:
        content = skill_md.read_text(encoding='utf-8')
        # 解析 YAML frontmatter
        if content.startswith('---'):
            lines = content.split('\n')
            if len(lines) > 1:
                yaml_section = []
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == '---':
                        break
                    yaml_section.append(line)
                
                # 简单解析 YAML（只支持 key: value 格式）
                for line in yaml_section:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key in METADATA_FIELDS:
                            metadata[key] = value
    except Exception as e:
        print(f"⚠️  警告: 无法解析 SKILL.md 元数据: {e}")
    
    return metadata


def create_manifest(
    skill_path: Path,
    files_included: List[Path],
    metadata: Dict[str, str],
    version: Optional[str] = None,
) -> Dict:
    """创建清单文件"""
    manifest = {
        'format_version': '1.0',
        'skill_name': skill_path.name,
        'version': version or metadata.get('version', '0.0.1'),
        'build_at': datetime.utcnow().isoformat() + 'Z',
        'metadata': metadata,
        'files': [],
        'statistics': {
            'total_files': len(files_included),
            'total_size': 0,
        }
    }
    
    total_size = 0
    for file_path in files_included:
        if file_path.is_file():
            file_size = file_path.stat().st_size
            total_size += file_size
            file_hash = calculate_file_hash(file_path)
            
            rel_path = file_path.relative_to(skill_path.parent)
            manifest['files'].append({
                'path': str(rel_path),
                'size': file_size,
                'hash': file_hash,
                'hash_algorithm': 'sha256',
            })
    
    manifest['statistics']['total_size'] = total_size
    return manifest


# ============================================================================
# 核心打包逻辑
# ============================================================================

class SkillPackager:
    """Skill 打包器"""
    
    def __init__(
        self,
        skill_path: Path,
        output_dir: Optional[Path] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        exclude_patterns: Optional[Set[str]] = None,
        include_manifest: bool = True,
        validate: bool = True,
    ):
        self.skill_path = Path(skill_path).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir else Path.cwd()
        self.version = version
        self.metadata = metadata or {}
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS.copy()
        self.include_manifest = include_manifest
        self.validate = validate and VALIDATION_AVAILABLE
        
        # 验证 skill 路径
        if not self.skill_path.exists():
            raise ValueError(f"Skill 路径不存在: {self.skill_path}")
        if not self.skill_path.is_dir():
            raise ValueError(f"Skill 路径不是目录: {self.skill_path}")
        
        skill_md = self.skill_path / 'SKILL.md'
        if not skill_md.exists():
            raise ValueError(f"SKILL.md 不存在: {skill_md}")
    
    def collect_files(self) -> List[Path]:
        """收集需要打包的文件"""
        files = []
        for file_path in self.skill_path.rglob('*'):
            if file_path.is_file():
                if not should_exclude_file(file_path, self.exclude_patterns):
                    files.append(file_path)
        return sorted(files)
    
    def package(self) -> Path:
        """执行打包"""
        print(f"📦 开始打包 Skill: {self.skill_path.name}")
        print(f"   路径: {self.skill_path}")
        print(f"   输出目录: {self.output_dir}\n")
        
        # 验证（如果启用）
        if self.validate:
            print("🔍 验证 Skill...")
            try:
                valid, message = validate_skill(self.skill_path)
                if not valid:
                    raise ValueError(f"验证失败: {message}")
                print(f"✅ {message}\n")
            except Exception as e:
                if VALIDATION_AVAILABLE:
                    raise
                else:
                    print(f"⚠️  跳过验证: {e}\n")
        
        # 收集文件
        print("📁 收集文件...")
        files = self.collect_files()
        print(f"   找到 {len(files)} 个文件\n")
        
        # 读取元数据
        skill_metadata = read_skill_metadata(self.skill_path)
        skill_metadata.update(self.metadata)
        
        # 确定版本号
        version = self.version or skill_metadata.get('version', '0.0.1')
        skill_metadata['version'] = version
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定输出文件名
        skill_name = self.skill_path.name
        if version and version != '0.0.1':
            output_filename = f"{skill_name}-{version}.skill"
        else:
            output_filename = f"{skill_name}.skill"
        output_file = self.output_dir / output_filename
        
        # 创建清单
        manifest = None
        if self.include_manifest:
            print("📋 生成清单文件...")
            manifest = create_manifest(self.skill_path, files, skill_metadata, version)
            print(f"   清单包含 {len(manifest['files'])} 个文件")
            print(f"   总大小: {manifest['statistics']['total_size'] / 1024:.2f} KB\n")
        
        # 创建 ZIP 文件
        print("🗜️  创建 .skill 文件...")
        try:
            with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加文件
                for file_path in files:
                    arcname = file_path.relative_to(self.skill_path.parent)
                    zipf.write(file_path, arcname)
                    print(f"   ✓ {arcname}")
                
                # 添加清单文件（如果启用）
                # 将 MANIFEST.json 放在 skill 目录内，与 SKILL.md 同级
                if manifest:
                    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
                    # 计算清单文件的 arcname：skill_name/MANIFEST.json
                    manifest_arcname = f"{self.skill_path.name}/{MANIFEST_FILE}"
                    zipf.writestr(manifest_arcname, manifest_json)
                    print(f"   ✓ {manifest_arcname}")
            
            file_size = output_file.stat().st_size
            print(f"\n✅ 打包成功!")
            print(f"   文件: {output_file}")
            print(f"   大小: {file_size / 1024:.2f} KB")
            if version:
                print(f"   版本: {version}")
            
            return output_file
            
        except Exception as e:
            print(f"\n❌ 打包失败: {e}")
            raise


# ============================================================================
# 命令行接口
# ============================================================================

def parse_metadata(metadata_str: str) -> Dict[str, str]:
    """解析元数据字符串（格式: key=value,key2=value2）"""
    metadata = {}
    if not metadata_str:
        return metadata
    
    for item in metadata_str.split(','):
        item = item.strip()
        if '=' in item:
            key, value = item.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            metadata[key] = value
    
    return metadata


def parse_exclude_patterns(patterns_str: str) -> Set[str]:
    """解析排除模式字符串（格式: pattern1,pattern2）"""
    if not patterns_str:
        return DEFAULT_EXCLUDE_PATTERNS.copy()
    
    patterns = set(DEFAULT_EXCLUDE_PATTERNS)
    for pattern in patterns_str.split(','):
        pattern = pattern.strip()
        if pattern:
            patterns.add(pattern)
    return patterns


def main():
    parser = argparse.ArgumentParser(
        description='工业级 Skill 打包工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'skill_path',
        type=str,
        help='Skill 目录路径'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出目录（默认：当前目录）'
    )
    parser.add_argument(
        '--version', '-v',
        type=str,
        default=None,
        help='版本号（如：1.0.0，默认：从 SKILL.md 读取或 0.0.0）'
    )
    parser.add_argument(
        '--metadata', '-m',
        type=str,
        default=None,
        help='元数据（格式：key=value,key2=value2）'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        default=None,
        help='排除模式（格式：pattern1,pattern2，会合并到默认排除列表）'
    )
    parser.add_argument(
        '--no-manifest',
        action='store_true',
        help='不包含清单文件'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='跳过验证步骤'
    )
    
    args = parser.parse_args()
    
    try:
        # 解析参数
        skill_path = Path(args.skill_path).resolve()
        output_dir = Path(args.output).resolve() if args.output else None
        metadata = parse_metadata(args.metadata) if args.metadata else {}
        exclude_patterns = parse_exclude_patterns(args.exclude) if args.exclude else None
        
        # 创建打包器
        packager = SkillPackager(
            skill_path=skill_path,
            output_dir=output_dir,
            version=args.version,
            metadata=metadata,
            exclude_patterns=exclude_patterns,
            include_manifest=not args.no_manifest,
            validate=not args.no_validate,
        )
        
        # 执行打包
        output_file = packager.package()
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
