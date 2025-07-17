'use client'

import { useState } from 'react'
import VideoGenerator from './components/VideoGenerator'
import BatchGenerator from './components/BatchGenerator'
import JobMonitor from './components/JobMonitor'
import VideoGallery from './components/VideoGallery'
import SystemStatus from './components/SystemStatus'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs'

export default function Home() {
  const [activeTab, setActiveTab] = useState('single')

  return (
    <div className="space-y-8">
      {/* System Status */}
      <SystemStatus />
      
      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="single">Single Video</TabsTrigger>
          <TabsTrigger value="batch">Batch Generation</TabsTrigger>
          <TabsTrigger value="jobs">Job Monitor</TabsTrigger>
          <TabsTrigger value="gallery">Video Gallery</TabsTrigger>
        </TabsList>
        
        <TabsContent value="single" className="space-y-6">
          <div className="card p-6">
            <h2 className="text-2xl font-bold mb-4">🚀 Quantum Video Generation</h2>
            <p className="text-gray-600 mb-6">
              Generate high-quality videos with AI-powered content selection and music synchronization.
              Uses cached captions for ultra-fast processing.
            </p>
            <VideoGenerator />
          </div>
        </TabsContent>
        
        <TabsContent value="batch" className="space-y-6">
          <div className="card p-6">
            <h2 className="text-2xl font-bold mb-4">📊 Batch Video Generation</h2>
            <p className="text-gray-600 mb-6">
              Generate multiple videos at once with different variations and scripts.
              Perfect for content creators who need volume.
            </p>
            <BatchGenerator />
          </div>
        </TabsContent>
        
        <TabsContent value="jobs" className="space-y-6">
          <div className="card p-6">
            <h2 className="text-2xl font-bold mb-4">⏱️ Job Monitor</h2>
            <p className="text-gray-600 mb-6">
              Track the progress of your video generation jobs in real-time.
            </p>
            <JobMonitor />
          </div>
        </TabsContent>
        
        <TabsContent value="gallery" className="space-y-6">
          <div className="card p-6">
            <h2 className="text-2xl font-bold mb-4">🎞️ Video Gallery</h2>
            <p className="text-gray-600 mb-6">
              Browse and download your generated videos.
            </p>
            <VideoGallery />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}