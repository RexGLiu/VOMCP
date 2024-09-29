from setuptools import setup

#####
# Base Environment
#####
setup(name='AdhocReasoningEnv',
      version='1.0.0',
      install_requires=['gym']
)

#####
# Paper Benchmark Environments
#####

setup(name='LaserTagEnv',
      version='1.0.0',
      install_requires=['gym','numpy']
)
