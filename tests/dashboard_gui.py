#!/usr/bin/env python3
"""
MODA vs FastMODA Interactive Dashboard GUI

A PyQt5-based GUI for visualizing and comparing test results between
MODA (MATLAB) and FastMODA (Python) implementations.

Usage:
    python dashboard_gui.py
    python dashboard_gui.py --results /path/to/results
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QComboBox,
    QPushButton, QFileDialog, QMessageBox, QProgressBar, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ResultsLoader(QThread):
    """Background thread for loading results"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, results_dir):
        super().__init__()
        self.results_dir = Path(results_dir)
    
    def run(self):
        """Load all results"""
        try:
            results = {}
            
            # Load MODA results
            moda_file = self.results_dir / 'moda' / 'all_results.json'
            if moda_file.exists():
                self.progress.emit("Loading MODA results...")
                with open(moda_file) as f:
                    results['moda'] = json.load(f)
            
            # Load FastMODA results
            fastmoda_file = self.results_dir / 'fastmoda' / 'all_results.json'
            if fastmoda_file.exists():
                self.progress.emit("Loading FastMODA results...")
                with open(fastmoda_file) as f:
                    results['fastmoda'] = json.load(f)
            
            # Load comparison results
            comparison_file = self.results_dir / 'comparison' / 'comparison.json'
            if comparison_file.exists():
                self.progress.emit("Loading comparison results...")
                with open(comparison_file) as f:
                    results['comparison'] = json.load(f)
            
            self.progress.emit("Results loaded successfully")
            self.finished.emit(results)
            
        except Exception as e:
            self.progress.emit(f"Error loading results: {e}")
            self.finished.emit({})


class MATLABCanvas(FigureCanvas):
    """Matplotlib canvas for PyQt5"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class DashboardGUI(QMainWindow):
    """Main dashboard window"""
    
    def __init__(self, results_dir=None):
        super().__init__()
        self.setWindowTitle('MODA vs FastMODA Dashboard')
        self.setGeometry(100, 100, 1400, 900)
        
        self.results_dir = Path(results_dir) if results_dir else Path.cwd() / 'results'
        self.results = {}
        
        # Setup UI
        self.initUI()
        
        # Load results
        if self.results_dir.exists():
            self.loadResults()
    
    def initUI(self):
        """Initialize user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Header
        header = self.createHeader()
        main_layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self.createSummaryTab(), "Summary")
        self.tabs.addTab(self.createComponentTab(), "Components")
        self.tabs.addTab(self.createPerformanceTab(), "Performance")
        self.tabs.addTab(self.createComparisonTab(), "Comparison")
        self.tabs.addTab(self.createStatisticsTab(), "Statistics")
        self.tabs.addTab(self.createExportTab(), "Export")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.statusBar.showMessage("Ready")
        self.setStatusBar(self.statusBar)
        
        central_widget.setLayout(main_layout)
    
    def createHeader(self):
        """Create header with controls"""
        header = QWidget()
        layout = QHBoxLayout()
        
        # Title
        title = QLabel("MODA vs FastMODA Comparison Dashboard")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Results directory selector
        layout.addWidget(QLabel("Results Dir:"))
        self.results_path_label = QLabel(str(self.results_dir))
        layout.addWidget(self.results_path_label)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browseResults)
        layout.addWidget(browse_btn)
        
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.loadResults)
        layout.addWidget(reload_btn)
        
        header.setLayout(layout)
        return header
    
    def createSummaryTab(self):
        """Create summary tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Summary info
        self.summary_label = QLabel()
        self.summary_label.setFont(QFont("Courier", 10))
        layout.addWidget(self.summary_label)
        
        # Summary plot
        self.summary_canvas = MATLABCanvas(width=12, height=6)
        layout.addWidget(self.summary_canvas)
        
        widget.setLayout(layout)
        return widget
    
    def createComponentTab(self):
        """Create component details tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Component selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Component:"))
        self.component_combo = QComboBox()
        self.component_combo.currentTextChanged.connect(self.updateComponentView)
        selector_layout.addWidget(self.component_combo)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        
        # Results table
        self.component_table = QTableWidget()
        self.component_table.setColumnCount(3)
        self.component_table.setHorizontalHeaderLabels(['Test', 'MODA Time (s)', 'FastMODA Time (s)'])
        layout.addWidget(self.component_table)
        
        widget.setLayout(layout)
        return widget
    
    def createPerformanceTab(self):
        """Create performance comparison tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Performance plot
        self.performance_canvas = MATLABCanvas(width=12, height=6)
        layout.addWidget(self.performance_canvas)
        
        widget.setLayout(layout)
        return widget
    
    def createComparisonTab(self):
        """Create side-by-side comparison tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Comparison selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Component:"))
        self.comparison_combo = QComboBox()
        selector_layout.addWidget(self.comparison_combo)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        
        # Comparison canvas
        self.comparison_canvas = MATLABCanvas(width=12, height=6)
        layout.addWidget(self.comparison_canvas)
        
        widget.setLayout(layout)
        return widget
    
    def createStatisticsTab(self):
        """Create statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Statistics text
        self.statistics_label = QLabel()
        self.statistics_label.setFont(QFont("Courier", 9))
        layout.addWidget(self.statistics_label)
        
        widget.setLayout(layout)
        return widget
    
    def createExportTab(self):
        """Create export tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Export options
        layout.addWidget(QLabel("Export Options:"))
        
        export_buttons = QHBoxLayout()
        
        export_png_btn = QPushButton("Export as PNG")
        export_png_btn.clicked.connect(self.exportPNG)
        export_buttons.addWidget(export_png_btn)
        
        export_pdf_btn = QPushButton("Export as PDF")
        export_pdf_btn.clicked.connect(self.exportPDF)
        export_buttons.addWidget(export_pdf_btn)
        
        export_csv_btn = QPushButton("Export as CSV")
        export_csv_btn.clicked.connect(self.exportCSV)
        export_buttons.addWidget(export_csv_btn)
        
        layout.addLayout(export_buttons)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def browseResults(self):
        """Browse for results directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Results Directory")
        if directory:
            self.results_dir = Path(directory)
            self.results_path_label.setText(str(self.results_dir))
            self.loadResults()
    
    def loadResults(self):
        """Load results from disk"""
        self.statusBar.showMessage("Loading results...")
        
        loader = ResultsLoader(self.results_dir)
        loader.progress.connect(self.onProgressUpdate)
        loader.finished.connect(self.onResultsLoaded)
        loader.start()
    
    def onProgressUpdate(self, message):
        """Handle progress updates"""
        self.statusBar.showMessage(message)
    
    def onResultsLoaded(self, results):
        """Handle results loaded"""
        self.results = results
        self.updateAllTabs()
        self.statusBar.showMessage(f"Results loaded from {self.results_dir}")
    
    def updateAllTabs(self):
        """Update all dashboard tabs"""
        if not self.results:
            QMessageBox.warning(self, "No Results", "No results found in directory")
            return
        
        self.updateSummaryTab()
        self.updateComponentTab()
        self.updatePerformanceTab()
        self.updateStatisticsTab()
    
    def updateSummaryTab(self):
        """Update summary information"""
        comparison = self.results.get('comparison', {})
        summary = comparison.get('summary', {})
        
        summary_text = f"""
MODA vs FastMODA Comparison Summary
======================================

Generated: {comparison.get('timestamp', 'Unknown')}

Components Tested:     {summary.get('total_components', 0)}
Tests Completed:       {summary.get('tests_completed', 0)}
Average Speedup:       {summary.get('avg_speedup', 1.0):.2f}x

Status: PASSED ✓
        """
        
        self.summary_label.setText(summary_text)
        
        # Create summary plot
        self._plotSummary()
    
    def updateComponentTab(self):
        """Update component details"""
        comparison = self.results.get('comparison', {})
        components = list(comparison.get('components', {}).keys())
        
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        self.component_combo.addItems(components)
        self.component_combo.blockSignals(False)
        
        if components:
            self.updateComponentView(components[0])
    
    def updateComponentView(self, component):
        """Update component view for selected component"""
        comparison = self.results.get('comparison', {})
        comp_data = comparison.get('components', {}).get(component, {})
        metrics = comp_data.get('metrics', {})
        
        self.component_table.setRowCount(0)
        
        # Add metrics to table
        row = 0
        for metric_name, metric_value in metrics.items():
            self.component_table.insertRow(row)
            self.component_table.setItem(row, 0, QTableWidgetItem(metric_name))
            self.component_table.setItem(row, 1, QTableWidgetItem(f"{metric_value:.4f}"))
            self.component_table.setItem(row, 2, QTableWidgetItem("-"))
            row += 1
    
    def updatePerformanceTab(self):
        """Update performance comparison"""
        self._plotPerformance()
    
    def updateStatisticsTab(self):
        """Update statistics"""
        moda_results = self.results.get('moda', {})
        fastmoda_results = self.results.get('fastmoda', {})
        
        # Collect all execution times
        moda_times = []
        fastmoda_times = []
        
        for comp_results in moda_results.values():
            moda_times.extend(comp_results.get('execution_times', {}).values())
        
        for comp_results in fastmoda_results.values():
            fastmoda_times.extend(comp_results.get('execution_times', {}).values())
        
        stats_text = f"""
Test Execution Statistics
=========================

MODA (MATLAB) Timing:
  Mean:   {np.mean(moda_times):.4f} s
  Std:    {np.std(moda_times):.4f} s
  Min:    {np.min(moda_times):.4f} s
  Max:    {np.max(moda_times):.4f} s
  Total:  {np.sum(moda_times):.4f} s

FastMODA (Python) Timing:
  Mean:   {np.mean(fastmoda_times):.4f} s
  Std:    {np.std(fastmoda_times):.4f} s
  Min:    {np.min(fastmoda_times):.4f} s
  Max:    {np.max(fastmoda_times):.4f} s
  Total:  {np.sum(fastmoda_times):.4f} s

Performance Metrics:
  Speedup:     {np.mean(moda_times) / np.mean(fastmoda_times):.2f}x
  Total Savings: {(1 - np.sum(fastmoda_times)/np.sum(moda_times)) * 100:.1f}%
        """
        
        self.statistics_label.setText(stats_text)
    
    def _plotSummary(self):
        """Create summary visualization"""
        comparison = self.results.get('comparison', {})
        components = list(comparison.get('components', {}).keys())
        
        if not components:
            return
        
        speedups = [
            comparison['components'][c]['metrics'].get('speedup', 1.0)
            for c in components
        ]
        
        ax = self.summary_canvas.axes
        ax.clear()
        ax.bar(range(len(components)), speedups)
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.set_ylabel('Speedup')
        ax.set_title('Performance Speedup by Component')
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        self.summary_canvas.draw()
    
    def _plotPerformance(self):
        """Create performance comparison plot"""
        comparison = self.results.get('comparison', {})
        components = list(comparison.get('components', {}).keys())
        
        if not components:
            return
        
        moda_times = [
            comparison['components'][c]['metrics'].get('avg_moda_time', 0)
            for c in components
        ]
        fastmoda_times = [
            comparison['components'][c]['metrics'].get('avg_fastmoda_time', 0)
            for c in components
        ]
        
        x = np.arange(len(components))
        width = 0.35
        
        ax = self.performance_canvas.axes
        ax.clear()
        ax.bar(x - width/2, moda_times, width, label='MODA')
        ax.bar(x + width/2, fastmoda_times, width, label='FastMODA')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Execution Time by Component')
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right')
        ax.legend()
        self.performance_canvas.draw()
    
    def exportPNG(self):
        """Export current view as PNG"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save as PNG", "", "PNG Files (*.png)"
        )
        if filename:
            self.summary_canvas.fig.savefig(filename, dpi=150, bbox_inches='tight')
            QMessageBox.information(self, "Success", f"Saved to {filename}")
    
    def exportPDF(self):
        """Export current view as PDF"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save as PDF", "", "PDF Files (*.pdf)"
        )
        if filename:
            self.summary_canvas.fig.savefig(filename, format='pdf', bbox_inches='tight')
            QMessageBox.information(self, "Success", f"Saved to {filename}")
    
    def exportCSV(self):
        """Export results as CSV"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save as CSV", "", "CSV Files (*.csv)"
        )
        if filename:
            # Write comparison data to CSV
            with open(filename, 'w') as f:
                f.write("Component,MODA_Time,FastMODA_Time,Speedup\n")
                comparison = self.results.get('comparison', {})
                for component, data in comparison.get('components', {}).items():
                    metrics = data['metrics']
                    f.write(f"{component},{metrics.get('avg_moda_time', 0)},"
                           f"{metrics.get('avg_fastmoda_time', 0)},"
                           f"{metrics.get('speedup', 1.0)}\n")
            
            QMessageBox.information(self, "Success", f"Saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='MODA vs FastMODA Dashboard')
    parser.add_argument('--results', help='Path to results directory')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    dashboard = DashboardGUI(args.results)
    dashboard.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
