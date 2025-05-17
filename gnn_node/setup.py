from setuptools import setup, find_packages
import os

package_name = 'gnn_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),  # 全てのパッケージを自動検出
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), ['config/model.param.yaml'])
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torch-geometric',
        'rclpy',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Node to subscribe to GraphData and prepare for GNN processing',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graph_subscriber_node = gnn_node.graph_subscriber_node:main',
        ],
    },
)
