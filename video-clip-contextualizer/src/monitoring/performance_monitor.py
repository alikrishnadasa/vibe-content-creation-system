import time
import psutil
import torch
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import json
from collections import deque

from ..config import get_config


@dataclass
class PerformanceMetrics:
    timestamp: datetime
    processing_time: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    confidence_score: Optional[float] = None
    video_segments: int = 0
    text_segments: int = 0
    matches_found: int = 0


@dataclass
class SystemStats:
    avg_processing_time: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_confidence: float = 0.0
    uptime: float = 0.0
    peak_memory_usage: float = 0.0
    peak_gpu_usage: float = 0.0


class PerformanceMonitor:
    """Monitors system performance and provides analytics."""
    
    def __init__(self, max_history: int = 1000):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.stats = SystemStats()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.active_requests = {}
        
        # GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        
        # Background monitoring
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self.monitoring_interval = 30  # seconds
        
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background system monitoring."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self._monitor_thread.start()
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_system(self):
        """Background thread for system monitoring."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {str(e)}")
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        gpu_usage = None
        gpu_memory_usage = None
        
        if self.gpu_available:
            try:
                # Get GPU statistics
                gpu_usage = self._get_gpu_usage()
                gpu_memory_usage = self._get_gpu_memory_usage()
            except Exception as e:
                self.logger.warning(f"Failed to get GPU metrics: {str(e)}")
        
        # Update peak usage
        if memory_usage > self.stats.peak_memory_usage:
            self.stats.peak_memory_usage = memory_usage
        
        if gpu_usage and gpu_usage > self.stats.peak_gpu_usage:
            self.stats.peak_gpu_usage = gpu_usage
        
        # Log high resource usage
        if memory_usage > 90:
            self.logger.warning(f"High memory usage: {memory_usage:.1f}%")
        
        if gpu_usage and gpu_usage > 90:
            self.logger.warning(f"High GPU usage: {gpu_usage:.1f}%")
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        if not self.gpu_available:
            return None
        
        try:
            # Use nvidia-ml-py if available, fallback to torch
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except ImportError:
            # Fallback: estimate from memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_cached = torch.cuda.memory_reserved(0)
                if memory_cached > 0:
                    return (memory_allocated / memory_cached) * 100
        except Exception:
            pass
        
        return None
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage percentage."""
        if not self.gpu_available:
            return None
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (mem_info.used / mem_info.total) * 100
        except ImportError:
            # Fallback: use torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_reserved = torch.cuda.max_memory_reserved(0)
                if memory_reserved > 0:
                    return (memory_allocated / memory_reserved) * 100
        except Exception:
            pass
        
        return None
    
    def start_request(self, request_id: str) -> str:
        """Start tracking a request."""
        self.active_requests[request_id] = {
            "start_time": time.time(),
            "timestamp": datetime.now()
        }
        return request_id
    
    def end_request(self, request_id: str, success: bool = True, 
                   confidence_score: float = None, video_segments: int = 0,
                   text_segments: int = 0, matches_found: int = 0):
        """End tracking a request and record metrics."""
        if request_id not in self.active_requests:
            self.logger.warning(f"Request {request_id} not found in active requests")
            return
        
        request_info = self.active_requests.pop(request_id)
        processing_time = time.time() - request_info["start_time"]
        
        # Get current system metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        gpu_usage = self._get_gpu_usage()
        gpu_memory_usage = self._get_gpu_memory_usage()
        
        # Create metrics record
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            processing_time=processing_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            confidence_score=confidence_score,
            video_segments=video_segments,
            text_segments=text_segments,
            matches_found=matches_found
        )
        
        self.metrics_history.append(metrics)
        
        # Update statistics
        self.stats.total_requests += 1
        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
        
        # Update averages
        self._update_averages()
        
        # Check performance against targets
        self._check_performance_targets(metrics)
        
        self.logger.info(f"Request {request_id} completed in {processing_time:.3f}s")
    
    def _update_averages(self):
        """Update running averages."""
        if not self.metrics_history:
            return
        
        recent_metrics = list(self.metrics_history)
        
        # Update processing time average
        processing_times = [m.processing_time for m in recent_metrics]
        self.stats.avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Update confidence average
        confidences = [m.confidence_score for m in recent_metrics if m.confidence_score is not None]
        if confidences:
            self.stats.avg_confidence = sum(confidences) / len(confidences)
        
        # Update uptime
        self.stats.uptime = (datetime.now() - self.start_time).total_seconds()
    
    def _check_performance_targets(self, metrics: PerformanceMetrics):
        """Check if performance meets targets and log warnings."""
        targets = self.config.performance
        
        # Check processing time
        if metrics.processing_time > (targets.max_processing_time_ms / 1000):
            self.logger.warning(
                f"Processing time {metrics.processing_time:.3f}s exceeds target "
                f"{targets.max_processing_time_ms/1000:.3f}s"
            )
        
        # Check confidence
        if metrics.confidence_score and metrics.confidence_score < targets.target_accuracy:
            self.logger.warning(
                f"Confidence {metrics.confidence_score:.3f} below target "
                f"{targets.target_accuracy}"
            )
        
        # Check memory usage
        if metrics.memory_usage > 90:
            self.logger.warning(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        # Check GPU usage
        if metrics.gpu_usage and metrics.gpu_usage < targets.gpu_utilization_target * 100:
            self.logger.info(
                f"GPU utilization {metrics.gpu_usage:.1f}% below target "
                f"{targets.gpu_utilization_target * 100:.1f}%"
            )
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            "system_stats": {
                "avg_processing_time": self.stats.avg_processing_time,
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "success_rate": (
                    self.stats.successful_requests / self.stats.total_requests 
                    if self.stats.total_requests > 0 else 0
                ),
                "avg_confidence": self.stats.avg_confidence,
                "uptime": self.stats.uptime,
                "peak_memory_usage": self.stats.peak_memory_usage,
                "peak_gpu_usage": self.stats.peak_gpu_usage
            },
            "current_system": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "gpu_usage": self._get_gpu_usage(),
                "gpu_memory_usage": self._get_gpu_memory_usage(),
                "active_requests": len(self.active_requests)
            },
            "targets": {
                "max_processing_time_ms": self.config.performance.max_processing_time_ms,
                "target_accuracy": self.config.performance.target_accuracy,
                "gpu_utilization_target": self.config.performance.gpu_utilization_target
            }
        }
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_metrics = [
            {
                "timestamp": m.timestamp.isoformat(),
                "processing_time": m.processing_time,
                "cpu_usage": m.cpu_usage,
                "memory_usage": m.memory_usage,
                "gpu_usage": m.gpu_usage,
                "gpu_memory_usage": m.gpu_memory_usage,
                "confidence_score": m.confidence_score,
                "video_segments": m.video_segments,
                "text_segments": m.text_segments,
                "matches_found": m.matches_found
            }
            for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        return filtered_metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        stats = self.get_current_stats()
        recent_metrics = self.get_metrics_history(hours=1)
        
        # Calculate trends
        if len(recent_metrics) >= 2:
            processing_times = [m["processing_time"] for m in recent_metrics]
            confidences = [m["confidence_score"] for m in recent_metrics if m["confidence_score"]]
            
            processing_trend = "improving" if processing_times[-1] < processing_times[0] else "declining"
            confidence_trend = "improving" if confidences and confidences[-1] > confidences[0] else "declining"
        else:
            processing_trend = "stable"
            confidence_trend = "stable"
        
        return {
            "summary": stats,
            "trends": {
                "processing_time": processing_trend,
                "confidence": confidence_trend
            },
            "recent_metrics": recent_metrics,
            "alerts": self._get_performance_alerts(),
            "recommendations": self._get_performance_recommendations()
        }
    
    def _get_performance_alerts(self) -> List[str]:
        """Get current performance alerts."""
        alerts = []
        
        # Check recent metrics for issues
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 requests
        
        if recent_metrics:
            avg_processing_time = sum(m.processing_time for m in recent_metrics) / len(recent_metrics)
            target_time = self.config.performance.max_processing_time_ms / 1000
            
            if avg_processing_time > target_time:
                alerts.append(f"Average processing time ({avg_processing_time:.3f}s) exceeds target ({target_time:.3f}s)")
            
            confidences = [m.confidence_score for m in recent_metrics if m.confidence_score]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence < self.config.performance.target_accuracy:
                    alerts.append(f"Average confidence ({avg_confidence:.3f}) below target ({self.config.performance.target_accuracy})")
        
        # Check system resources
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 90:
            alerts.append(f"High memory usage: {memory_usage:.1f}%")
        
        gpu_memory = self._get_gpu_memory_usage()
        if gpu_memory and gpu_memory > 90:
            alerts.append(f"High GPU memory usage: {gpu_memory:.1f}%")
        
        return alerts
    
    def _get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        # Analyze patterns
        if len(self.metrics_history) >= 10:
            recent_metrics = list(self.metrics_history)[-10:]
            
            # Check for consistently high processing times
            processing_times = [m.processing_time for m in recent_metrics]
            avg_time = sum(processing_times) / len(processing_times)
            target_time = self.config.performance.max_processing_time_ms / 1000
            
            if avg_time > target_time * 1.5:
                recommendations.append("Consider reducing video clip duration or increasing overlap to reduce processing load")
            
            # Check GPU utilization
            gpu_usages = [m.gpu_usage for m in recent_metrics if m.gpu_usage is not None]
            if gpu_usages:
                avg_gpu = sum(gpu_usages) / len(gpu_usages)
                target_gpu = self.config.performance.gpu_utilization_target * 100
                
                if avg_gpu < target_gpu * 0.5:
                    recommendations.append("GPU utilization is low; consider increasing batch size or using CPU processing")
                elif avg_gpu > 95:
                    recommendations.append("GPU utilization is very high; consider reducing batch size")
            
            # Check confidence patterns
            confidences = [m.confidence_score for m in recent_metrics if m.confidence_score]
            if confidences:
                low_confidence_ratio = sum(1 for c in confidences if c < 0.5) / len(confidences)
                if low_confidence_ratio > 0.3:
                    recommendations.append("High ratio of low-confidence matches; consider adjusting matching strategy or preprocessing")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable ranges")
        
        return recommendations
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "system_stats": self.get_current_stats(),
                "metrics_history": self.get_metrics_history(hours=24)
            }
            
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            raise