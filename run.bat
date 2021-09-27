rmdir /s/q build
rmdir /s/q dist
rmdir /s/q mksc.egg-info
pip uninstall mksc
python setup.py sdist bdist_wheel
python -m twine upload  dist/*
rem python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
rem pip install mksc==2.0.0 -i https://pypi.org/simple