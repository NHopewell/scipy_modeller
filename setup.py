from distutils.core import setup


def readme():
    """
    Import the README.md Markdown file and try 
    to convert it to RST format.
    """
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()
        

setup(
    name='scipy-modeler',
    version='0.1',
    description="""Implement production Machine Learning/Data """
                """Science workflows with scipy-modeler""",
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    # Substitute <github_account> with the name of your GitHub account
    url='https://github.com/ .....',
    author='Nick Hopewell',
    author_email='nicholashopewell@gmail.com',  
    license='MIT',
    packages=['scipy-modeler'],
)

