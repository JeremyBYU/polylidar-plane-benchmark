from setuptools import setup, find_packages
setup(
    name="polylidar_plane_benchmark",
    version="0.0.1",
    packages=['polylidar_plane_benchmark'],
    scripts=[],

    install_requires=['Click', 'numpy', 'open3d', 'colorama','shapely','scipy', 'matplotlib','descartes', 'pandas', 'seaborn', 'tqdm',
     'pypcd@https://api.github.com/repos/jeremybyu/pypcd/tarball/'],

    entry_points='''
        [console_scripts]
        ppb=polylidar_plane_benchmark.scripts.cli:cli
    ''',

    # metadata to display on PyPI
    author="Jeremy Castagno",
    author_email="jdcasta@umich.edu",
    description="Polylidar Plane Benchmark",
    license="MIT",
    keywords="concave hull benchmark",
    url="https://github.com/JeremyBYU/polylidar-plane-benchmark",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/JeremyBYU/polylidar-plane-benchmark/issues",
    }
)