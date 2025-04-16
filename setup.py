"""
@author: skitsanos
"""

from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="unified-llm-client",
    version="0.2.1",
    author="skitsanos",
    author_email="info@skitsanos.com",
    description="A unified async client for interacting with multiple LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skitsanos/unified-llm-client",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.22.0",
        "openai>=1.28.0",
        "python-dotenv>=1.0.0"
    ],
    keywords="llm, openai, anthropic, gpt, claude, ai, machine learning, ollama",
)
