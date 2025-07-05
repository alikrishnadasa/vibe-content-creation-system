"""
Precise Synchronization Engine
Frame-perfect timing for captions, audio, and visual elements
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import time
import math


@dataclass
class SyncEvent:
    """A synchronization event with precise timing"""
    name: str
    start_time: float
    duration: float
    event_type: str  # 'caption', 'beat', 'visual', 'transition'
    priority: int = 1  # Higher = more important
    data: Dict[str, Any] = None
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    def overlaps_with(self, other: 'SyncEvent', tolerance: float = 0.001) -> bool:
        """Check if this event overlaps with another within tolerance"""
        return not (self.end_time <= other.start_time + tolerance or 
                   other.end_time <= self.start_time + tolerance)


@dataclass 
class TimingConstraint:
    """Constraint for timing relationships between events"""
    event1_id: str
    event2_id: str
    constraint_type: str  # 'before', 'after', 'simultaneous', 'min_gap', 'max_gap'
    value: float = 0.0  # For gap constraints, the gap duration
    tolerance: float = 0.001


class PreciseSyncEngine:
    """
    Frame-perfect synchronization engine
    
    Features:
    - Sub-millisecond timing precision
    - Event conflict resolution
    - Adaptive timing adjustment
    - Performance optimization
    """
    
    def __init__(self, frame_rate: float = 30.0, config: Dict = None):
        """Initialize sync engine"""
        self.frame_rate = frame_rate
        self.frame_duration = 1.0 / frame_rate
        self.config = config or {}
        
        # Event management
        self.events: List[SyncEvent] = []
        self.constraints: List[TimingConstraint] = []
        
        # Timing precision
        self.timing_precision = self.config.get('timing_precision', 0.001)  # 1ms
        self.max_adjustment_iterations = self.config.get('max_adjustments', 100)
        
        # Performance tracking
        self.stats = {
            'events_processed': 0,
            'adjustments_made': 0,
            'conflicts_resolved': 0,
            'average_sync_accuracy': 0.0,
            'last_sync_time': 0.0
        }
    
    def add_event(self, event: SyncEvent) -> str:
        """Add a synchronization event"""
        # Generate unique ID if not set
        if not hasattr(event, 'id'):
            event.id = f"{event.event_type}_{len(self.events)}_{int(time.time() * 1000)}"
        
        self.events.append(event)
        return event.id
    
    def add_constraint(self, constraint: TimingConstraint):
        """Add a timing constraint between events"""
        self.constraints.append(constraint)
    
    def add_caption_events(self, captions: List[Dict]) -> List[str]:
        """Add caption events to the timeline"""
        event_ids = []
        
        for i, caption in enumerate(captions):
            event = SyncEvent(
                name=f"caption_{i}",
                start_time=caption['start_time'],
                duration=caption['end_time'] - caption['start_time'],
                event_type='caption',
                priority=2,
                data=caption
            )
            event_id = self.add_event(event)
            event_ids.append(event_id)
        
        return event_ids
    
    def add_beat_events(self, beats: List[float], intensity: List[float] = None) -> List[str]:
        """Add beat synchronization events"""
        event_ids = []
        
        for i, beat_time in enumerate(beats):
            beat_intensity = intensity[i] if intensity and i < len(intensity) else 0.5
            
            event = SyncEvent(
                name=f"beat_{i}",
                start_time=beat_time,
                duration=0.05,  # Brief duration for beat markers
                event_type='beat',
                priority=3,
                data={'intensity': beat_intensity, 'beat_index': i}
            )
            event_id = self.add_event(event)
            event_ids.append(event_id)
        
        return event_ids
    
    def add_visual_events(self, visual_cues: List[Dict]) -> List[str]:
        """Add visual synchronization events"""
        event_ids = []
        
        for i, cue in enumerate(visual_cues):
            event = SyncEvent(
                name=f"visual_{i}",
                start_time=cue.get('time', 0.0),
                duration=cue.get('duration', 0.1),
                event_type='visual',
                priority=1,
                data=cue
            )
            event_id = self.add_event(event)
            event_ids.append(event_id)
        
        return event_ids
    
    def snap_to_frame_boundaries(self):
        """Snap all events to frame boundaries for perfect video sync"""
        for event in self.events:
            # Snap start time to nearest frame
            frame_number = round(event.start_time / self.frame_duration)
            event.start_time = frame_number * self.frame_duration
            
            # Adjust duration to maintain end time or snap end time too
            if self.config.get('snap_duration', False):
                duration_frames = round(event.duration / self.frame_duration)
                event.duration = duration_frames * self.frame_duration
    
    def detect_conflicts(self) -> List[Tuple[SyncEvent, SyncEvent]]:
        """Detect timing conflicts between events"""
        conflicts = []
        
        for i, event1 in enumerate(self.events):
            for event2 in self.events[i+1:]:
                # Check for overlapping high-priority events
                if (event1.priority > 1 and event2.priority > 1 and 
                    event1.event_type == event2.event_type and
                    event1.overlaps_with(event2)):
                    conflicts.append((event1, event2))
        
        return conflicts
    
    def resolve_conflicts(self) -> int:
        """Resolve timing conflicts using priority and adjustment"""
        conflicts = self.detect_conflicts()
        adjustments_made = 0
        
        for event1, event2 in conflicts:
            # Higher priority event keeps its timing
            if event1.priority > event2.priority:
                dominant, subordinate = event1, event2
            elif event2.priority > event1.priority:
                dominant, subordinate = event2, event1
            else:
                # Same priority - adjust the later one
                if event1.start_time <= event2.start_time:
                    dominant, subordinate = event1, event2
                else:
                    dominant, subordinate = event2, event1
            
            # Move subordinate event to avoid conflict
            min_start_time = dominant.end_time + self.timing_precision
            if subordinate.start_time < min_start_time:
                subordinate.start_time = min_start_time
                adjustments_made += 1
        
        self.stats['adjustments_made'] += adjustments_made
        self.stats['conflicts_resolved'] += len(conflicts)
        
        return adjustments_made
    
    def apply_constraints(self) -> int:
        """Apply timing constraints and make necessary adjustments"""
        adjustments_made = 0
        
        for constraint in self.constraints:
            event1 = self._find_event_by_id(constraint.event1_id)
            event2 = self._find_event_by_id(constraint.event2_id)
            
            if not event1 or not event2:
                continue
            
            if constraint.constraint_type == 'before':
                # Event1 must end before event2 starts
                if event1.end_time > event2.start_time - constraint.tolerance:
                    event2.start_time = event1.end_time + constraint.tolerance
                    adjustments_made += 1
            
            elif constraint.constraint_type == 'after':
                # Event1 must start after event2 ends
                if event1.start_time < event2.end_time + constraint.tolerance:
                    event1.start_time = event2.end_time + constraint.tolerance
                    adjustments_made += 1
            
            elif constraint.constraint_type == 'simultaneous':
                # Events should start at the same time
                time_diff = abs(event1.start_time - event2.start_time)
                if time_diff > constraint.tolerance:
                    # Adjust lower priority event
                    if event1.priority >= event2.priority:
                        event2.start_time = event1.start_time
                    else:
                        event1.start_time = event2.start_time
                    adjustments_made += 1
            
            elif constraint.constraint_type == 'min_gap':
                # Minimum gap between events
                actual_gap = event2.start_time - event1.end_time
                if actual_gap < constraint.value:
                    event2.start_time = event1.end_time + constraint.value
                    adjustments_made += 1
            
            elif constraint.constraint_type == 'max_gap':
                # Maximum gap between events
                actual_gap = event2.start_time - event1.end_time
                if actual_gap > constraint.value:
                    event2.start_time = event1.end_time + constraint.value
                    adjustments_made += 1
        
        return adjustments_made
    
    def _find_event_by_id(self, event_id: str) -> Optional[SyncEvent]:
        """Find event by ID"""
        for event in self.events:
            if hasattr(event, 'id') and event.id == event_id:
                return event
        return None
    
    def optimize_timing(self) -> Dict[str, Any]:
        """Optimize timing for all events"""
        start_time = time.time()
        
        # Sort events by start time
        self.events.sort(key=lambda e: e.start_time)
        
        # Iterative optimization
        total_adjustments = 0
        iteration = 0
        
        while iteration < self.max_adjustment_iterations:
            # Apply frame boundary snapping if enabled
            if self.config.get('frame_snapping', True):
                self.snap_to_frame_boundaries()
            
            # Resolve conflicts
            conflict_adjustments = self.resolve_conflicts()
            
            # Apply constraints
            constraint_adjustments = self.apply_constraints()
            
            total_adjustments += conflict_adjustments + constraint_adjustments
            
            # Stop if no more adjustments needed
            if conflict_adjustments == 0 and constraint_adjustments == 0:
                break
            
            iteration += 1
        
        # Calculate timing accuracy
        accuracy = self._calculate_timing_accuracy()
        
        optimization_time = time.time() - start_time
        
        # Update statistics
        self.stats.update({
            'events_processed': len(self.events),
            'adjustments_made': total_adjustments,
            'average_sync_accuracy': accuracy,
            'last_sync_time': optimization_time
        })
        
        return {
            'iterations': iteration,
            'total_adjustments': total_adjustments,
            'timing_accuracy': accuracy,
            'optimization_time': optimization_time,
            'events_count': len(self.events),
            'constraints_count': len(self.constraints)
        }
    
    def _calculate_timing_accuracy(self) -> float:
        """Calculate overall timing accuracy"""
        if not self.events:
            return 100.0
        
        # Check how many events are perfectly aligned to frame boundaries
        perfectly_aligned = 0
        
        for event in self.events:
            frame_time = round(event.start_time / self.frame_duration) * self.frame_duration
            if abs(event.start_time - frame_time) <= self.timing_precision:
                perfectly_aligned += 1
        
        return (perfectly_aligned / len(self.events)) * 100.0
    
    def get_events_at_time(self, time: float, tolerance: float = 0.001) -> List[SyncEvent]:
        """Get all events active at a specific time"""
        active_events = []
        
        for event in self.events:
            if (event.start_time - tolerance <= time <= event.end_time + tolerance):
                active_events.append(event)
        
        return active_events
    
    def get_events_in_range(self, start_time: float, end_time: float) -> List[SyncEvent]:
        """Get all events within a time range"""
        events_in_range = []
        
        for event in self.events:
            if not (event.end_time < start_time or event.start_time > end_time):
                events_in_range.append(event)
        
        return events_in_range
    
    def export_timeline(self) -> Dict[str, Any]:
        """Export synchronized timeline for rendering"""
        # Sort events by start time
        sorted_events = sorted(self.events, key=lambda e: e.start_time)
        
        timeline = {
            'frame_rate': self.frame_rate,
            'frame_duration': self.frame_duration,
            'total_duration': max(e.end_time for e in sorted_events) if sorted_events else 0,
            'events': [],
            'frame_events': {},  # Events organized by frame number
            'statistics': self.stats
        }
        
        # Export events
        for event in sorted_events:
            timeline['events'].append({
                'id': getattr(event, 'id', f"{event.event_type}_{event.start_time}"),
                'name': event.name,
                'type': event.event_type,
                'start_time': event.start_time,
                'end_time': event.end_time,
                'duration': event.duration,
                'priority': event.priority,
                'data': event.data or {}
            })
            
            # Organize by frame for efficient lookup during rendering
            start_frame = int(event.start_time / self.frame_duration)
            end_frame = int(event.end_time / self.frame_duration)
            
            for frame in range(start_frame, end_frame + 1):
                if frame not in timeline['frame_events']:
                    timeline['frame_events'][frame] = []
                timeline['frame_events'][frame].append(event)
        
        return timeline
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronization engine statistics"""
        return {
            **self.stats,
            'total_events': len(self.events),
            'total_constraints': len(self.constraints),
            'frame_rate': self.frame_rate,
            'timing_precision': self.timing_precision
        }


# Integration functions
async def sync_captions_with_audio(caption_engine, audio_data, sync_engine):
    """Synchronize captions with audio using precise timing"""
    # This would integrate with the caption engine
    pass


async def sync_visuals_with_beats(visual_data, beat_data, sync_engine):
    """Synchronize visual elements with musical beats"""
    # This would integrate with the beat sync engine
    pass


# Test function
async def test_precise_sync():
    """Test the precise synchronization engine"""
    print("🧪 Testing Precise Sync Engine...")
    
    # Initialize engine
    sync_engine = PreciseSyncEngine(frame_rate=30.0, config={
        'frame_snapping': True,
        'timing_precision': 0.001
    })
    
    # Test 1: Add caption events
    print("\n1️⃣ Testing Caption Events")
    captions = [
        {'start_time': 0.5, 'end_time': 1.2, 'text': 'Hello'},
        {'start_time': 1.5, 'end_time': 2.1, 'text': 'world'},
        {'start_time': 2.3, 'end_time': 3.0, 'text': 'sync'}
    ]
    
    caption_ids = sync_engine.add_caption_events(captions)
    print(f"   Added {len(caption_ids)} caption events")
    
    # Test 2: Add beat events
    print("\n2️⃣ Testing Beat Events")
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    intensity = [0.8, 0.6, 0.9, 0.7, 0.8, 0.5, 0.9]
    
    beat_ids = sync_engine.add_beat_events(beats, intensity)
    print(f"   Added {len(beat_ids)} beat events")
    
    # Test 3: Add constraints
    print("\n3️⃣ Testing Constraints")
    if len(caption_ids) >= 2:
        constraint = TimingConstraint(
            event1_id=caption_ids[0],
            event2_id=caption_ids[1],
            constraint_type='min_gap',
            value=0.1  # Minimum 100ms gap
        )
        sync_engine.add_constraint(constraint)
        print("   Added min_gap constraint between first two captions")
    
    # Test 4: Optimize timing
    print("\n4️⃣ Testing Timing Optimization")
    result = sync_engine.optimize_timing()
    print(f"   Optimization completed in {result['iterations']} iterations")
    print(f"   Total adjustments: {result['total_adjustments']}")
    print(f"   Timing accuracy: {result['timing_accuracy']:.1f}%")
    print(f"   Optimization time: {result['optimization_time']:.4f}s")
    
    # Test 5: Export timeline
    print("\n5️⃣ Testing Timeline Export")
    timeline = sync_engine.export_timeline()
    print(f"   Timeline duration: {timeline['total_duration']:.2f}s")
    print(f"   Total events: {len(timeline['events'])}")
    print(f"   Frame events organized for {len(timeline['frame_events'])} frames")
    
    # Test 6: Query events
    print("\n6️⃣ Testing Event Queries")
    events_at_1s = sync_engine.get_events_at_time(1.0)
    print(f"   Events at 1.0s: {len(events_at_1s)}")
    
    events_in_range = sync_engine.get_events_in_range(0.5, 2.0)
    print(f"   Events in range 0.5-2.0s: {len(events_in_range)}")
    
    # Test 7: Statistics
    print("\n7️⃣ Testing Statistics")
    stats = sync_engine.get_statistics()
    print(f"   Events processed: {stats['events_processed']}")
    print(f"   Conflicts resolved: {stats['conflicts_resolved']}")
    print(f"   Average accuracy: {stats['average_sync_accuracy']:.1f}%")
    
    print("\n✅ All sync engine tests completed!")


if __name__ == "__main__":
    asyncio.run(test_precise_sync()) 