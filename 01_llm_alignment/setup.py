from setuptools import setup, find_packages
import os

# README 파일 읽기
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# requirements.txt에서 의존성 읽기 (선택사항)
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="llm-alignment",
    version="0.1.0",
    description="LLM Alignment Training Pipeline following Anthropic's HHH approach",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/llm-alignment-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python 3.10+ 권장 (3.11이 최적)
    python_requires=">=3.10",
    
    install_requires=[
        # PyTorch 2.x (Python 3.10+ 최적화)
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torchaudio>=2.1.0",
        
        # Transformers (최신 버전)
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "tokenizers>=0.14.0",
        "accelerate>=0.24.0",
        
        # 핵심 라이브러리들
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        
        # 유틸리티
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        "wandb>=0.16.0",
        "huggingface-hub>=0.17.0",
        "safetensors>=0.4.0",
        
        # 타입 관련 (Python 3.10+ 기능 활용)
        "typing-extensions>=4.8.0",
    ],
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",  # 미래 대비
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # 선택적 의존성 그룹
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "ipywidgets>=8.0.0",
        ],
        "evaluation": [
            "rouge-score>=0.1.2",
            "sacrebleu>=2.3.0",
            "bert-score>=0.3.13",
            "nltk>=3.8.0",
        ],
        "distributed": [
            "deepspeed>=0.9.0",
            "fairscale>=0.4.13",
        ],
        "monitoring": [
            "tensorboard>=2.13.0",
            "mlflow>=2.4.0",
        ],
        "all": [
            # dev 의존성
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
            # notebooks 의존성
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "ipywidgets>=8.0.0",
            # evaluation 의존성
            "rouge-score>=0.1.2",
            "sacrebleu>=2.3.0",
            "bert-score>=0.3.13",
            "nltk>=3.8.0",
            # distributed 의존성
            "deepspeed>=0.9.0",
            "fairscale>=0.4.13",
            # monitoring 의존성
            "tensorboard>=2.13.0",
            "mlflow>=2.4.0",
        ]
    },
    
    # 키워드
    keywords="llm, language model, alignment, rlhf, anthropic, transformer, pytorch",
    
    # 엔트리포인트 (CLI 명령어로 사용할 수 있음)
    entry_points={
        "console_scripts": [
            "llm-pretrain=scripts.train_pretrain:main",
            "llm-sft=scripts.train_sft:main",
            "llm-rlhf=scripts.train_rlhf:main",
            "llm-eval=scripts.evaluate:main",
            "llm-infer=scripts.inference:main",
        ],
    },
    
    # 포함할 추가 파일들
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    # 프로젝트 URL들
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-alignment-project/issues",
        "Source": "https://github.com/yourusername/llm-alignment-project",
        "Documentation": "https://github.com/yourusername/llm-alignment-project/blob/main/README.md",
    },
)