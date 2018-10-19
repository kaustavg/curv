from setuptools import setup

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='curv',
	version='0.1',
	description='Manipulate Bayesian Networks of Continuous Random Variables',
	long_description=readme(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	url='http://github.com/kaustavg/curv',
	author='Kaustav Gopinathan',
	author_email='kaustav.gopinathan@gmail.com',
	license='MIT',
	packages=['curv'],
	install_requires=[
		'numpy',
		'scipy',
		'matplotlib',
	],
	include_package_data=True,
	zip_safe=False)