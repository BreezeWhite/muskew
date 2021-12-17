import setuptools


with open("README.md") as red:
    ldest = red.read()

with open("requirements.txt") as req_file:
    req = req_file.read()

setuptools.setup(
    name='muskew',
    version='0.1.0',
    author='BreezeWhite',
    author_email='miyasihta2010@tuta.io',
    description='Muskew helps deskew/dewarp the curved music sheet photo with AI-aidded approach.',
    long_description=ldest,
    long_description_content_type='text/markdown',
    license='MIT',
    license_files=('LICENSE',),
    url='https://github.com/BreezeWhite/music-sheet-deskewing',
    packages=setuptools.find_packages(),
    package_data={
        '': [
                'checkpoint/metadata.pkl',
                'checkpoint/model.onnx'
            ]
    },
    install_requires=req,
    entry_points={'console_scripts': ['muskew = muskew.main:main']},
)
