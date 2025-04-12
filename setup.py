from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mkt_sim_cl",
    version="0.1.0",
    author="Mario Curcio",
    author_email="mario.curcio@example.com",
    description="Sistema di simulazione del mercato azionario con trading algoritmico",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mkt_sim_cl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mkt_sim=mkt_sim_cl.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mkt_sim_cl": [
            "data/*",
            "models/*",
            "config/*",
            "templates/*",
            "static/*",
        ]
    },
) 