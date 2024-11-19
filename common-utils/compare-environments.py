import subprocess
import sys
import os
from datetime import datetime

REPORT_PATH = 'common-utils/environment_comparison.txt'

def get_venv_info():
    """Get virtual environment name and path"""
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        venv_name = os.path.basename(venv_path)
        return {
            'name': venv_name,
            'path': venv_path,
            'active': True
        }
    return {
        'name': 'None',
        'path': 'None',
        'active': False
    }

def get_python_version(python_path):
    """Get Python version using specified Python executable"""
    try:
        result = subprocess.run([python_path, '--version'], 
                              capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error getting Python version: {str(e)}"

def get_pip_list(python_path):
    """Get list of installed packages using specified Python executable"""
    try:
        result = subprocess.run([python_path, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        return result.stdout.strip().split('\n')[2:]  # Skip header rows
    except Exception as e:
        return [f"Error getting pip list: {str(e)}"]

def get_package_dict(packages_list):
    """Convert pip list output to dictionary of package names and versions"""
    packages = {}
    for package in packages_list:
        try:
            name, version = package.split()[:2]  # Split and take first two elements
            packages[name.lower()] = version
        except:
            continue
    return packages

def generate_report():
    """Generate comparison report between virtual and global environments"""
    # Get virtual environment information
    venv_info = get_venv_info()
    
    # Get virtual environment Python path
    venv_python = sys.executable
    
    # Get global Python path
    global_python = subprocess.run(['which', 'python3'], 
                                 capture_output=True, 
                                 text=True).stdout.strip()
    
    # Get package information for both environments
    venv_packages = get_package_dict(get_pip_list(venv_python))
    global_packages = get_package_dict(get_pip_list(global_python))
    
    # Prepare report
    report = []
    report.append("Python Environment Comparison Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add virtual environment information
    report.append("\nVirtual Environment Information:")
    report.append(f"Name: {venv_info['name']}")
    report.append(f"Path: {venv_info['path']}")
    report.append(f"Status: {'Active' if venv_info['active'] else 'Inactive'}")
    
    report.append("\n1. Python Versions:")
    report.append(f"Virtual Environment: {get_python_version(venv_python)}")
    report.append(f"Global Environment: {get_python_version(global_python)}")
    
    report.append("\n2. Package Version Comparison:")
    report.append(f"{'Package':<30} {'Virtual Env':<15} {'Global':<15} {'Status':<20}")
    report.append("-" * 80)
    
    # Combine all package names from both environments
    all_packages = sorted(set(list(venv_packages.keys()) + list(global_packages.keys())))
    
    for package in all_packages:
        venv_version = venv_packages.get(package, "Not installed")
        global_version = global_packages.get(package, "Not installed")
        
        status = ""
        if venv_version == global_version:
            status = "Same version"
        elif venv_version == "Not installed":
            status = "Global only"
        elif global_version == "Not installed":
            status = "Virtual env only"
        else:
            status = "Version mismatch"
        
        report.append(f"{package:<30} {venv_version:<15} {global_version:<15} {status:<20}")
    
    # Write report to file
    report_path = REPORT_PATH
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    return report_path

if __name__ == "__main__":
    try:
        if not os.environ.get('VIRTUAL_ENV'):
            print("Error: This script must be run from within a virtual environment")
            sys.exit(1)
            
        report_path = generate_report()
        print(f"\nReport generated successfully: {report_path}")
        print("\nSummary of report:")
        with open(report_path, 'r') as f:
            print(f.read())
            
    except Exception as e:
        print(f"Error generating report: {str(e)}")