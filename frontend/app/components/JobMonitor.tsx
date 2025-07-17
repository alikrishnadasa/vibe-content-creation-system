'use client'

import { useState, useEffect } from 'react'
import { Clock, CheckCircle, XCircle, RefreshCw, Trash2, Download } from 'lucide-react'
import { formatTimeAgo, formatDuration } from '../lib/utils'

interface Job {
  job_id: string
  status: string
  progress: number
  message: string
  created_at: string
  completed_at?: string
  result?: any
  request_type: string
  processing_time?: number
}

export default function JobMonitor() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    fetchJobs()
    
    let interval: NodeJS.Timeout
    if (autoRefresh) {
      interval = setInterval(fetchJobs, 3000) // Refresh every 3 seconds
    }
    
    return () => clearInterval(interval)
  }, [autoRefresh])

  const fetchJobs = async () => {
    try {
      const response = await fetch('/api/jobs')
      const data = await response.json()
      setJobs(data.jobs)
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
    } finally {
      setLoading(false)
    }
  }

  const deleteJob = async (jobId: string) => {
    try {
      await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' })
      setJobs(jobs.filter(job => job.job_id !== jobId))
    } catch (error) {
      console.error('Failed to delete job:', error)
    }
  }

  const downloadResult = (jobId: string) => {
    window.open(`/api/download/${jobId}`, '_blank')
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-600" />
      case 'processing':
        return <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
      default:
        return <Clock className="w-5 h-5 text-yellow-600" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-50 border-green-200'
      case 'failed':
        return 'bg-red-50 border-red-200'
      case 'processing':
        return 'bg-blue-50 border-blue-200'
      default:
        return 'bg-yellow-50 border-yellow-200'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        <span className="ml-2 text-gray-600">Loading jobs...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={fetchJobs}
            className="btn-secondary flex items-center"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </button>
          
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="mr-2"
            />
            <span className="text-sm text-gray-600">Auto-refresh</span>
          </label>
        </div>

        <div className="text-sm text-gray-500">
          {jobs.length} total jobs
        </div>
      </div>

      {/* Jobs List */}
      {jobs.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Clock className="w-12 h-12 mx-auto mb-4 text-gray-300" />
          <p>No jobs found</p>
          <p className="text-sm">Start generating videos to see jobs here</p>
        </div>
      ) : (
        <div className="space-y-4">
          {jobs.map((job) => (
            <div
              key={job.job_id}
              className={`card p-4 ${getStatusColor(job.status)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center mb-2">
                    {getStatusIcon(job.status)}
                    <span className="ml-2 font-medium">
                      {job.request_type === 'batch' ? '📊 Batch Generation' : '🎬 Single Video'}
                    </span>
                    <span className="ml-2 text-sm text-gray-500">
                      {formatTimeAgo(job.created_at)}
                    </span>
                  </div>

                  <p className="text-sm text-gray-600 mb-2">
                    {job.message}
                  </p>

                  {/* Progress Bar for Processing Jobs */}
                  {job.status === 'processing' && (
                    <div className="mb-3">
                      <div className="flex justify-between text-xs text-gray-600 mb-1">
                        <span>Progress</span>
                        <span>{Math.round(job.progress)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="progress-bar h-2 rounded-full transition-all duration-300"
                          style={{ width: `${job.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {/* Job Details */}
                  <div className="flex items-center space-x-4 text-xs text-gray-500">
                    <span>ID: {job.job_id.slice(0, 8)}...</span>
                    {job.processing_time && (
                      <span>Time: {formatDuration(job.processing_time)}</span>
                    )}
                    {job.completed_at && (
                      <span>Completed: {formatTimeAgo(job.completed_at)}</span>
                    )}
                  </div>

                  {/* Result Summary */}
                  {job.status === 'completed' && job.result && (
                    <div className="mt-3 p-2 bg-white bg-opacity-50 rounded text-xs">
                      {job.request_type === 'batch' ? (
                        <span>Generated {job.result.total_videos} videos</span>
                      ) : (
                        <span>
                          Video generated • {(job.result.file_size / (1024 * 1024)).toFixed(1)}MB
                          {job.result.target_achieved && ' • ⚡ Target achieved'}
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center space-x-2 ml-4">
                  {job.status === 'completed' && job.result?.output_path && (
                    <button
                      onClick={() => downloadResult(job.job_id)}
                      className="p-2 text-green-600 hover:bg-green-100 rounded-lg"
                      title="Download"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  )}
                  
                  <button
                    onClick={() => deleteJob(job.job_id)}
                    className="p-2 text-red-600 hover:bg-red-100 rounded-lg"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}