import os
import sys
import subprocess
import shutil
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
import platform

class CMakeMultiGPUBuild(_build_ext):
    """
    Enhanced CMake build process for multi-GPU support
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmake_source_dir = None
        self.cmake_build_dir = None
        
    def run(self):
        """Main build process"""
        try:
            self.check_dependencies()
            self.setup_directories()
            self.configure_cmake()
            self.run_cmake_build()
            self.copy_extensions()
        except Exception as e:
            print(f"Build failed: {e}")
            raise
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        # Check for CMake
        try:
            result = subprocess.run(['cmake', '--version'], 
                                  capture_output=True, text=True, check=True)
            cmake_version = result.stdout.split('\n')[0]
            print(f"Found {cmake_version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("CMake is required but not found. Please install CMake >= 3.18")
        
        # Check for CUDA
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, check=True)
            print("CUDA compiler found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: CUDA compiler not found. Multi-GPU features may not work.")
        
        # Check for pybind11
        try:
            import pybind11
            print(f"Found pybind11 version {pybind11.__version__}")
        except ImportError:
            raise RuntimeError("pybind11 is required but not found. Please install pybind11")
    
    def setup_directories(self):
        """Setup build directories"""
        self.cmake_source_dir = os.path.abspath(os.path.dirname(__file__))
        self.cmake_build_dir = os.path.join(self.build_temp, 'cmake_build')
        
        # Create build directory
        os.makedirs(self.cmake_build_dir, exist_ok=True)
        print(f"Build directory: {self.cmake_build_dir}")
        print(f"Source directory: {self.cmake_source_dir}")
    
    def configure_cmake(self):
        """Configure CMake build"""
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(self.build_lib)}',
            f'-DCMAKE_BUILD_TYPE={"Debug" if self.debug else "Release"}',
            '-DENABLE_MULTIGPU=ON',
        ]
        
        # Add pybind11 path if available
        try:
            import pybind11
            pybind11_dir = pybind11.get_cmake_dir()
            cmake_args.append(f'-Dpybind11_DIR={pybind11_dir}')
            print(f"Using pybind11 from: {pybind11_dir}")
        except:
            print("Using system pybind11")
        
        # Platform-specific configuration
        if platform.system() == "Windows":
            cmake_args.extend([
                '-DCMAKE_GENERATOR_PLATFORM=x64',
            ])
        
        # Use enhanced CMakeLists.txt
        cmake_file = os.path.join(self.cmake_source_dir, 'CMakeLists_MultiGPU.txt')
        if os.path.exists(cmake_file):
            # Copy the enhanced CMakeLists.txt
            shutil.copy(cmake_file, os.path.join(self.cmake_source_dir, 'CMakeLists.txt'))
            print("Using enhanced CMakeLists.txt for multi-GPU build")
        
        print("Configuring CMake...")
        print(f"CMake args: {' '.join(cmake_args)}")
        
        result = subprocess.run(['cmake', self.cmake_source_dir] + cmake_args, 
                              cwd=self.cmake_build_dir, check=True)
    
    def run_cmake_build(self):
        """Run the CMake build process"""
        build_args = []
        
        # Determine number of parallel jobs
        if hasattr(os, 'cpu_count'):
            parallel_jobs = os.cpu_count()
        else:
            parallel_jobs = 2
        
        build_args.extend(['--parallel', str(parallel_jobs)])
        
        print(f"Building with {parallel_jobs} parallel jobs...")
        
        subprocess.run(['cmake', '--build', '.'] + build_args, 
                      cwd=self.cmake_build_dir, check=True)
    
    def copy_extensions(self):
        """Copy built extensions to the appropriate location"""
        # The CMake build should output to the correct location
        # but we can add additional copying logic here if needed
        print("Build completed successfully!")
        
        # List built files
        if os.path.exists(self.build_lib):
            built_files = list(Path(self.build_lib).rglob('*.so'))
            built_files.extend(Path(self.build_lib).rglob('*.pyd'))
            built_files.extend(Path(self.build_lib).rglob('*.dll'))
            
            if built_files:
                print("Built extensions:")
                for file in built_files:
                    print(f"  {file}")
            else:
                print("Warning: No extension files found in build directory")

def get_requirements():
    """Get package requirements"""
    requirements = [
        'numpy>=1.18.0',
        'pybind11>=2.6.0',
    ]
    
    # Add optional requirements for enhanced functionality
    optional_requirements = [
        'matplotlib>=3.0.0',  # For plotting and visualization
        'scipy>=1.5.0',       # For scientific computing
    ]
    
    return requirements

def get_cuda_extensions():
    """Get CUDA extension configuration if available"""
    extensions = []
    
    # Check if CUDA is available
    try:
        subprocess.run(['nvcc', '--version'], 
                      capture_output=True, check=True)
        cuda_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        cuda_available = False
    
    if cuda_available:
        # Multi-GPU extension
        multigpu_sources = [
            'src/multiGPUWrapper.cpp',
            'src/multiGPUMemory.cu',
            'src/multiGPUControl.cu', 
            'src/multiGPUKernels.cu',
            'src/goldCode.cpp',
            'src/gpuCode.cu',
            'src/gpuControl.cu',
        ]
        
        extensions.append(Extension(
            'libspinChainMultiGPU',
            sources=multigpu_sources,
            include_dirs=[
                'src/',
            ],
            libraries=['cudart', 'cublas', 'curand', 'cufft'],
            language='c++',
        ))
        
        # Single-GPU extension for compatibility
        singlegpu_sources = [
            'src/pybindWrapper.cpp',
            'src/goldCode.cpp',
            'src/gpuCode.cu',
            'src/gpuControl.cu',
        ]
        
        extensions.append(Extension(
            'libspinChain',
            sources=singlegpu_sources,
            include_dirs=[
                'src/',
            ],
            libraries=['cudart', 'cufft'],
            language='c++',
        ))
    
    return extensions

def main():
    """Main setup function"""
    
    # Read version from file if available
    version = "2.0.0-multigpu"
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            version = f.read().strip()
    
    # Read long description from README
    long_description = ""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            long_description = f.read()
    
    setup(
        name='spinChainMultiGPU',
        version=version,
        author='Multi-GPU Enhancement Team',
        author_email='enhanced@example.com',
        description='Multi-GPU enhanced Bethe Functions for XXZ Spin Chain',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/your-repo/Bethe-Functions',
        
        packages=find_packages(),
        
        # Use CMake build system
        ext_modules=[Extension('dummy', sources=[])],  # Placeholder
        cmdclass={'build_ext': CMakeMultiGPUBuild},
        
        # Requirements
        install_requires=get_requirements(),
        python_requires='>=3.7',
        
        # Classifiers
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Physics',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: C++',
            'Programming Language :: CUDA',
        ],
        
        # Package data
        include_package_data=True,
        zip_safe=False,
        
        # Entry points for command-line tools
        entry_points={
            'console_scripts': [
                'spinchain-benchmark=spinChain.benchmark:main',
                'spinchain-info=spinChain.info:main',
            ],
        },
        
        # Additional metadata
        keywords='physics quantum mechanics bethe ansatz spin chain multi-gpu cuda',
        project_urls={
            'Bug Reports': 'https://github.com/your-repo/Bethe-Functions/issues',
            'Source': 'https://github.com/your-repo/Bethe-Functions',
            'Documentation': 'https://bethe-functions.readthedocs.io/',
        },
    )

if __name__ == '__main__':
    main()