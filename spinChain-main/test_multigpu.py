#!/usr/bin/env python3
"""
Multi-GPU Performance Testing and Validation Suite for Bethe Functions

This script provides comprehensive testing and benchmarking for the multi-GPU
implementation of Bethe function calculations.
"""

import sys
import time
import numpy as np
import argparse
from typing import List, Dict, Tuple, Any
import json
import os

try:
    import libspinChainMultiGPU as multigpu
    MULTIGPU_AVAILABLE = True
except ImportError:
    print("Warning: Multi-GPU library not available")
    MULTIGPU_AVAILABLE = False

try:
    import libspinChain as singlegpu
    SINGLEGPU_AVAILABLE = True
except ImportError:
    print("Warning: Single-GPU library not available")
    SINGLEGPU_AVAILABLE = False

class MultiGPUTester:
    """Comprehensive testing suite for multi-GPU Bethe functions"""
    
    def __init__(self):
        self.results = {
            'system_info': {},
            'correctness_tests': {},
            'performance_tests': {},
            'scalability_tests': {},
            'memory_tests': {}
        }
        
        if MULTIGPU_AVAILABLE:
            self.multigpu_chain = multigpu.MultiGPUSpinChain()
            self.system_info = self._get_system_info()
        else:
            print("Multi-GPU functionality not available")
            self.multigpu_chain = None
            self.system_info = {}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        info = {}
        
        if self.multigpu_chain:
            gpu_info = self.multigpu_chain.get_gpu_info()
            memory_info = self.multigpu_chain.get_memory_info()
            
            info.update({
                'gpu_info': gpu_info,
                'memory_info': memory_info,
                'available_gpus': self.multigpu_chain.get_available_gpus()
            })
        
        return info
    
    def generate_test_data(self, nBasis: int, nUp: int) -> Tuple[List, List, List, List, complex]:
        """Generate test data for Bethe function calculations"""
        
        # Generate Bethe roots (complex numbers)
        allBetheRoots = []
        for i in range(nBasis):
            for j in range(nUp):
                # Create complex roots with some physical meaning
                real_part = 0.5 + 0.1 * (i + j)
                imag_part = 0.1 * (i - j)
                allBetheRoots.append(complex(real_part, imag_part))
        
        # Generate configurations (particle positions)
        allConfigs = []
        config_base = list(range(nUp))
        for i in range(nBasis):
            # Generate different configurations
            config = [(config_base[j] + i) % (nUp + 2) for j in range(nUp)]
            allConfigs.extend(config)
        
        # Generate Gaudin determinants
        allGaudinDets = []
        for i in range(nBasis):
            det_value = complex(1.0 + 0.01 * i, 0.01 * i)
            allGaudinDets.append(det_value)
        
        # Generate permutation sigma
        sigma = []
        nPerm = min(120, max(6, nUp * 6))  # Approximate factorial, limited for testing
        for perm in range(nPerm):
            perm_indices = [(perm + j) % nUp for j in range(nUp)]
            sigma.extend(perm_indices)
        
        # Anisotropy parameter
        delta = complex(0.5, 0.0)
        
        return allBetheRoots, allConfigs, allGaudinDets, sigma, delta
    
    def test_correctness(self, test_sizes: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Test correctness by comparing single-GPU and multi-GPU results"""
        print("Running correctness tests...")
        
        correctness_results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': []
        }
        
        if not (MULTIGPU_AVAILABLE and SINGLEGPU_AVAILABLE):
            print("Cannot run correctness tests - both single and multi-GPU libraries required")
            return correctness_results
        
        for nBasis, nUp in test_sizes:
            print(f"  Testing size: nBasis={nBasis}, nUp={nUp}")
            
            test_result = {
                'nBasis': nBasis,
                'nUp': nUp,
                'passed': False,
                'error': None,
                'max_difference': None
            }
            
            try:
                # Generate test data
                allBetheRoots, allConfigs, allGaudinDets, sigma, delta = self.generate_test_data(nBasis, nUp)
                
                # Run single-GPU computation
                single_start = time.time()
                single_result = singlegpu.gpu_compute_basis_transform_single(
                    allBetheRoots, allConfigs, allGaudinDets, sigma, delta
                )
                single_time = time.time() - single_start
                
                # Run multi-GPU computation
                multi_start = time.time()
                multi_result = self.multigpu_chain.compute_basis_transform(
                    allBetheRoots, allConfigs, allGaudinDets, sigma, delta
                )
                multi_time = time.time() - multi_start
                
                # Compare results
                e2c_single, c2e_single = single_result
                e2c_multi, c2e_multi = multi_result
                
                e2c_diff = np.max(np.abs(np.array(e2c_single) - np.array(e2c_multi)))
                c2e_diff = np.max(np.abs(np.array(c2e_single) - np.array(c2e_multi)))
                max_diff = max(e2c_diff, c2e_diff)
                
                # Check if results match within tolerance
                tolerance = 1e-10
                if max_diff < tolerance:
                    test_result['passed'] = True
                    correctness_results['tests_passed'] += 1
                    print(f"    PASSED (max diff: {max_diff:.2e})")
                else:
                    correctness_results['tests_failed'] += 1
                    print(f"    FAILED (max diff: {max_diff:.2e})")
                
                test_result['max_difference'] = float(max_diff)
                test_result['single_gpu_time'] = single_time
                test_result['multi_gpu_time'] = multi_time
                
            except Exception as e:
                test_result['error'] = str(e)
                correctness_results['tests_failed'] += 1
                print(f"    ERROR: {e}")
            
            correctness_results['test_details'].append(test_result)
        
        return correctness_results
    
    def test_performance(self, test_sizes: List[Tuple[int, int]], gpu_counts: List[int]) -> Dict[str, Any]:
        """Test performance across different problem sizes and GPU counts"""
        print("Running performance tests...")
        
        performance_results = {
            'test_configurations': [],
            'summary': {}
        }
        
        if not MULTIGPU_AVAILABLE:
            print("Multi-GPU not available for performance testing")
            return performance_results
        
        for nBasis, nUp in test_sizes:
            for gpu_count in gpu_counts:
                print(f"  Testing: nBasis={nBasis}, nUp={nUp}, GPUs={gpu_count}")
                
                config_result = {
                    'nBasis': nBasis,
                    'nUp': nUp,
                    'gpu_count': gpu_count,
                    'execution_time': None,
                    'throughput': None,
                    'gpu_utilization': [],
                    'memory_usage': [],
                    'error': None
                }
                
                try:
                    # Set up GPU configuration
                    available_gpus = self.multigpu_chain.get_available_gpus()
                    if gpu_count > len(available_gpus):
                        print(f"    Skipping: Only {len(available_gpus)} GPUs available")
                        continue
                    
                    gpu_ids = available_gpus[:gpu_count]
                    self.multigpu_chain.set_active_gpus(gpu_ids)
                    
                    # Generate test data
                    allBetheRoots, allConfigs, allGaudinDets, sigma, delta = self.generate_test_data(nBasis, nUp)
                    
                    # Run computation
                    start_time = time.time()
                    result = self.multigpu_chain.compute_basis_transform(
                        allBetheRoots, allConfigs, allGaudinDets, sigma, delta
                    )
                    execution_time = time.time() - start_time
                    
                    # Get performance metrics
                    metrics = self.multigpu_chain.get_performance_metrics()
                    
                    config_result.update({
                        'execution_time': execution_time,
                        'throughput': (nBasis * nBasis) / execution_time,  # Operations per second
                        'gpu_utilization': metrics.get('gpu_utilization', []),
                        'memory_usage': metrics.get('gpu_memory_usage_mb', [])
                    })
                    
                    print(f"    Completed in {execution_time:.2f}s")
                    
                except Exception as e:
                    config_result['error'] = str(e)
                    print(f"    ERROR: {e}")
                
                performance_results['test_configurations'].append(config_result)
        
        return performance_results
    
    def test_scalability(self, base_size: Tuple[int, int], scale_factors: List[int]) -> Dict[str, Any]:
        """Test how performance scales with problem size"""
        print("Running scalability tests...")
        
        scalability_results = {
            'base_size': base_size,
            'scale_factors': scale_factors,
            'scaling_data': []
        }
        
        if not MULTIGPU_AVAILABLE:
            print("Multi-GPU not available for scalability testing")
            return scalability_results
        
        base_nBasis, base_nUp = base_size
        
        for scale_factor in scale_factors:
            nBasis = int(base_nBasis * scale_factor)
            nUp = min(base_nUp, nBasis // 2)  # Ensure physical constraints
            
            print(f"  Testing scale factor {scale_factor}: nBasis={nBasis}, nUp={nUp}")
            
            scale_result = {
                'scale_factor': scale_factor,
                'nBasis': nBasis,
                'nUp': nUp,
                'execution_time': None,
                'memory_used': None,
                'error': None
            }
            
            try:
                # Generate test data
                allBetheRoots, allConfigs, allGaudinDets, sigma, delta = self.generate_test_data(nBasis, nUp)
                
                # Run computation
                start_time = time.time()
                result = self.multigpu_chain.compute_basis_transform(
                    allBetheRoots, allConfigs, allGaudinDets, sigma, delta
                )
                execution_time = time.time() - start_time
                
                # Get memory usage
                memory_info = self.multigpu_chain.get_memory_info()
                memory_used = memory_info.get('used_memory_mb', 0)
                
                scale_result.update({
                    'execution_time': execution_time,
                    'memory_used': memory_used
                })
                
                print(f"    Completed in {execution_time:.2f}s, used {memory_used:.1f}MB")
                
            except Exception as e:
                scale_result['error'] = str(e)
                print(f"    ERROR: {e}")
            
            scalability_results['scaling_data'].append(scale_result)
        
        return scalability_results
    
    def test_memory_limits(self) -> Dict[str, Any]:
        """Test memory management and limits"""
        print("Running memory limit tests...")
        
        memory_results = {
            'total_memory': 0,
            'available_memory': 0,
            'max_problem_size': None,
            'memory_efficiency': []
        }
        
        if not MULTIGPU_AVAILABLE:
            print("Multi-GPU not available for memory testing")
            return memory_results
        
        # Get initial memory info
        memory_info = self.multigpu_chain.get_memory_info()
        memory_results['total_memory'] = memory_info.get('total_memory_mb', 0)
        memory_results['available_memory'] = memory_info.get('available_memory_mb', 0)
        
        print(f"  Total GPU memory: {memory_results['total_memory']:.1f} MB")
        print(f"  Available memory: {memory_results['available_memory']:.1f} MB")
        
        # Test different problem sizes to find memory limits
        test_sizes = [(10, 3), (20, 4), (50, 5), (100, 6), (200, 7)]
        
        for nBasis, nUp in test_sizes:
            print(f"  Testing memory usage: nBasis={nBasis}, nUp={nUp}")
            
            try:
                # Generate test data
                allBetheRoots, allConfigs, allGaudinDets, sigma, delta = self.generate_test_data(nBasis, nUp)
                
                # Attempt computation
                result = self.multigpu_chain.compute_basis_transform(
                    allBetheRoots, allConfigs, allGaudinDets, sigma, delta
                )
                
                # Get memory usage after computation
                memory_info = self.multigpu_chain.get_memory_info()
                used_memory = memory_info.get('used_memory_mb', 0)
                
                efficiency = {
                    'nBasis': nBasis,
                    'nUp': nUp,
                    'memory_used_mb': used_memory,
                    'success': True
                }
                
                memory_results['memory_efficiency'].append(efficiency)
                memory_results['max_problem_size'] = (nBasis, nUp)
                
                print(f"    Success: {used_memory:.1f} MB used")
                
            except Exception as e:
                efficiency = {
                    'nBasis': nBasis,
                    'nUp': nUp,
                    'error': str(e),
                    'success': False
                }
                memory_results['memory_efficiency'].append(efficiency)
                print(f"    Failed: {e}")
                break
        
        return memory_results
    
    def run_all_tests(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete test suite"""
        print("Starting comprehensive multi-GPU test suite...")
        print("=" * 60)
        
        # System information
        print("System Information:")
        if self.system_info:
            print(f"  Available GPUs: {len(self.system_info.get('available_gpus', []))}")
            memory_info = self.system_info.get('memory_info', {})
            if memory_info:
                print(f"  Total GPU Memory: {memory_info.get('total_memory_mb', 0):.1f} MB")
        print()
        
        # Run tests
        test_results = {
            'system_info': self.system_info,
            'timestamp': time.time(),
            'test_config': test_config
        }
        
        if test_config.get('run_correctness', True):
            test_results['correctness_tests'] = self.test_correctness(
                test_config.get('correctness_sizes', [(10, 3), (20, 4)])
            )
            print()
        
        if test_config.get('run_performance', True):
            test_results['performance_tests'] = self.test_performance(
                test_config.get('performance_sizes', [(20, 4), (50, 5)]),
                test_config.get('gpu_counts', [1, 2, 4])
            )
            print()
        
        if test_config.get('run_scalability', True):
            test_results['scalability_tests'] = self.test_scalability(
                test_config.get('base_size', (20, 4)),
                test_config.get('scale_factors', [1, 2, 3, 4])
            )
            print()
        
        if test_config.get('run_memory', True):
            test_results['memory_tests'] = self.test_memory_limits()
            print()
        
        return test_results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save test results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filename}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of test results"""
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        # Correctness tests
        if 'correctness_tests' in results:
            correctness = results['correctness_tests']
            total_tests = correctness['tests_passed'] + correctness['tests_failed']
            print(f"Correctness Tests: {correctness['tests_passed']}/{total_tests} passed")
        
        # Performance tests
        if 'performance_tests' in results:
            perf_tests = results['performance_tests']['test_configurations']
            successful_tests = [t for t in perf_tests if t.get('execution_time') is not None]
            print(f"Performance Tests: {len(successful_tests)}/{len(perf_tests)} completed")
            
            if successful_tests:
                avg_time = np.mean([t['execution_time'] for t in successful_tests])
                print(f"  Average execution time: {avg_time:.2f}s")
        
        # Memory tests
        if 'memory_tests' in results:
            memory = results['memory_tests']
            max_size = memory.get('max_problem_size')
            if max_size:
                print(f"Maximum problem size: nBasis={max_size[0]}, nUp={max_size[1]}")
            
            total_memory = memory.get('total_memory', 0)
            available_memory = memory.get('available_memory', 0)
            print(f"GPU Memory: {available_memory:.1f}/{total_memory:.1f} MB available")

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Multi-GPU Bethe Functions Test Suite')
    
    parser.add_argument('--correctness', action='store_true', default=True,
                       help='Run correctness tests')
    parser.add_argument('--performance', action='store_true', default=True,
                       help='Run performance tests') 
    parser.add_argument('--scalability', action='store_true', default=True,
                       help='Run scalability tests')
    parser.add_argument('--memory', action='store_true', default=True,
                       help='Run memory tests')
    parser.add_argument('--output', '-o', default='multigpu_test_results.json',
                       help='Output file for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only')
    
    args = parser.parse_args()
    
    # Configure tests based on arguments
    if args.quick:
        test_config = {
            'run_correctness': args.correctness,
            'correctness_sizes': [(10, 3)],
            'run_performance': args.performance,
            'performance_sizes': [(20, 4)],
            'gpu_counts': [1, 2],
            'run_scalability': args.scalability,
            'base_size': (10, 3),
            'scale_factors': [1, 2],
            'run_memory': args.memory
        }
    else:
        test_config = {
            'run_correctness': args.correctness,
            'correctness_sizes': [(10, 3), (20, 4), (30, 5)],
            'run_performance': args.performance,
            'performance_sizes': [(20, 4), (50, 5), (100, 6)],
            'gpu_counts': [1, 2, 4, 8],
            'run_scalability': args.scalability,
            'base_size': (20, 4),
            'scale_factors': [1, 1.5, 2, 3, 4],
            'run_memory': args.memory
        }
    
    # Run tests
    tester = MultiGPUTester()
    results = tester.run_all_tests(test_config)
    
    # Save and display results
    tester.save_results(results, args.output)
    tester.print_summary(results)

if __name__ == '__main__':
    main()