"""Target-independent LLVM function discovery and flip-site detection."""

from karnage.scanner.scanner import FlipSite, FunctionSites, ScanResult, scan_binary

__all__ = ["scan_binary", "FlipSite", "FunctionSites", "ScanResult"]
