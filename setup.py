import setuptools

with open("README.md", "r", encoding="utf_8_sig") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mksc",
    version="3.0.0",
    author="wuhaoyu",
    author_email="wuhaoyu96@163.com",
    description="It is used to quickly build a scorecard project and dichotomy model package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HelloCoyen/mksc.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
    entry_points={'console_scripts': ['mksc = mksc.core.cmd:main']},
    include_package_data=True,
    install_requires=["numpy", "pandas", "sklearn", "featuretools", "statsmodels", "pandas_profiling"]
    )
