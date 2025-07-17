'use client'

import { useState, useEffect } from 'react'
import { Play, Download, Calendar, HardDrive, RefreshCw } from 'lucide-react'
import { formatFileSize, formatTimeAgo } from '../lib/utils'

interface Video {
  name: string
  size: number
  size_mb: number
  created: string
  path: string
}

export default function VideoGallery() {
  const [videos, setVideos] = useState<Video[]>([])
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState<'date' | 'size' | 'name'>('date')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')

  useEffect(() => {
    fetchVideos()
  }, [])

  useEffect(() => {
    sortVideos()
  }, [sortBy, sortOrder, videos])

  const fetchVideos = async () => {
    try {
      const response = await fetch('/api/outputs')
      const data = await response.json()
      setVideos(data.videos)
    } catch (error) {
      console.error('Failed to fetch videos:', error)
    } finally {
      setLoading(false)
    }
  }

  const sortVideos = () => {
    const sorted = [...videos].sort((a, b) => {
      let aVal: any, bVal: any
      
      switch (sortBy) {
        case 'date':
          aVal = new Date(a.created).getTime()
          bVal = new Date(b.created).getTime()
          break
        case 'size':
          aVal = a.size
          bVal = b.size
          break
        case 'name':
          aVal = a.name.toLowerCase()
          bVal = b.name.toLowerCase()
          break
        default:
          return 0
      }

      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1
      } else {
        return aVal < bVal ? 1 : -1
      }
    })

    setVideos(sorted)
  }

  const downloadVideo = (video: Video) => {
    // Create a download link
    const link = document.createElement('a')
    link.href = `/videos/${video.name}`
    link.download = video.name
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const getTotalSize = () => {
    return videos.reduce((total, video) => total + video.size, 0)
  }

  const getVideoType = (filename: string) => {
    if (filename.includes('_mixed_')) return 'Mixed Audio'
    if (filename.includes('_var')) return 'Variation'
    if (filename.includes('batch')) return 'Batch'
    return 'Standard'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        <span className="ml-2 text-gray-600">Loading videos...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between space-y-4 sm:space-y-0">
        <div className="flex items-center space-x-4">
          <button
            onClick={fetchVideos}
            className="btn-secondary flex items-center"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </button>
          
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-600">Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'date' | 'size' | 'name')}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              <option value="date">Date</option>
              <option value="size">Size</option>
              <option value="name">Name</option>
            </select>
            
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="text-sm text-primary-600 hover:text-primary-700"
            >
              {sortOrder === 'asc' ? '↑' : '↓'}
            </button>
          </div>
        </div>

        <div className="text-sm text-gray-500">
          {videos.length} videos • {formatFileSize(getTotalSize())} total
        </div>
      </div>

      {/* Gallery */}
      {videos.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Play className="w-12 h-12 mx-auto mb-4 text-gray-300" />
          <p>No videos found</p>
          <p className="text-sm">Generate some videos to see them here</p>
        </div>
      ) : (
        <div className="video-grid">
          {videos.map((video) => (
            <div key={video.name} className="card p-4 hover:shadow-md transition-shadow">
              {/* Video Thumbnail Placeholder */}
              <div className="aspect-[9/16] bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg mb-4 flex items-center justify-center border-2 border-dashed border-gray-200">
                <div className="text-center">
                  <Play className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-xs text-gray-500">2:3 Aspect Ratio</p>
                  <p className="text-xs text-gray-400">{video.size_mb}MB</p>
                </div>
              </div>

              {/* Video Info */}
              <div className="space-y-2">
                <h3 className="font-medium text-sm leading-tight" title={video.name}>
                  {video.name.length > 40 ? `${video.name.slice(0, 37)}...` : video.name}
                </h3>

                <div className="flex items-center text-xs text-gray-500 space-x-2">
                  <span className="flex items-center">
                    <Calendar className="w-3 h-3 mr-1" />
                    {formatTimeAgo(video.created)}
                  </span>
                  <span className="flex items-center">
                    <HardDrive className="w-3 h-3 mr-1" />
                    {video.size_mb}MB
                  </span>
                </div>

                <div className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
                  {getVideoType(video.name)}
                </div>

                {/* Actions */}
                <div className="flex space-x-2 pt-2">
                  <button
                    onClick={() => downloadVideo(video)}
                    className="flex-1 btn-primary text-xs py-2 flex items-center justify-center"
                  >
                    <Download className="w-3 h-3 mr-1" />
                    Download
                  </button>
                  
                  <button
                    onClick={() => window.open(`/videos/${video.name}`, '_blank')}
                    className="flex-1 btn-secondary text-xs py-2 flex items-center justify-center"
                  >
                    <Play className="w-3 h-3 mr-1" />
                    Preview
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Statistics */}
      {videos.length > 0 && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-800 mb-3">📊 Gallery Statistics</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-gray-500">Total Videos</div>
              <div className="font-medium">{videos.length}</div>
            </div>
            <div>
              <div className="text-gray-500">Total Size</div>
              <div className="font-medium">{formatFileSize(getTotalSize())}</div>
            </div>
            <div>
              <div className="text-gray-500">Average Size</div>
              <div className="font-medium">
                {formatFileSize(getTotalSize() / videos.length)}
              </div>
            </div>
            <div>
              <div className="text-gray-500">Latest</div>
              <div className="font-medium">
                {videos.length > 0 ? formatTimeAgo(videos[0].created) : 'N/A'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}