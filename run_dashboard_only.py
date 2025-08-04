#!/usr/bin/env python3
"""
Simple Dashboard Runner
Just runs the dashboard without the complex pipeline.
"""

import sys
import subprocess

def main():
    """Run just the dashboard"""
    print("🚀 Starting Sales Analytics Dashboard...")
    print("📱 Dashboard will open in your web browser")
    print("🔗 Local URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("  - Use the sidebar filters to explore different views")
    print("  - Navigate through tabs to see different analyses")
    print("  - Hover over charts for detailed information")
    print("  - Press Ctrl+C in terminal to stop the dashboard")
    
    try:
        # Start Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main() 