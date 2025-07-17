'use client'

import { useState } from 'react'
import { Layers, Clock, AlertTriangle } from 'lucide-react'

interface BatchJob {
  job_id: string
  status: string
  progress: number
  message: string
  processing_time?: number
  result?: {
    total_videos: number
    output_directory: string
    caption_style: string
  }
}

export default function BatchGenerator() {
  const [numVideos, setNumVideos] = useState(5)
  const [captionStyle, setCaptionStyle] = useState('default')
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentJob, setCurrentJob] = useState<BatchJob | null>(null)

  const captionStyles = [
    { value: 'default', label: 'Default (Recommended)', recommended: true },
    { value: 'tiktok', label: 'TikTok Style' },
    { value: 'cinematic', label: 'Cinematic' },
    { value: 'minimal', label: 'Minimal' },
    { value: 'youtube', label: 'YouTube' },
    { value: 'karaoke', label: 'Karaoke' }
  ]

  const fetchJobStatus = async (jobId: string) => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`)
      const job = await response.json()
      setCurrentJob(job)
      
      if (job.status === 'completed' || job.status === 'failed') {
        setIsGenerating(false)
      } else {
        // Continue polling
        setTimeout(() => fetchJobStatus(jobId), 2000)
      }
    } catch (error) {
      console.error('Failed to fetch job status:', error)
    }
  }

  const generateBatch = async () => {
    setIsGenerating(true)
    setCurrentJob(null)

    try {
      const response = await fetch('/api/generate-batch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_videos: numVideos,
          caption_style: captionStyle
        }),
      })

      const data = await response.json()
      if (response.ok) {
        setCurrentJob({
          job_id: data.job_id,
          status: 'pending',
          progress: 0,
          message: 'Starting batch generation...'
        })
        // Start polling
        setTimeout(() => fetchJobStatus(data.job_id), 1000)
      } else {
        throw new Error(data.detail || 'Batch generation failed')
      }
    } catch (error) {
      console.error('Failed to start batch generation:', error)
      setIsGenerating(false)
    }
  }

  const estimatedTime = numVideos * 60 // Rough estimate: 1 minute per video

  return (
    <div className="space-y-6">
      {/* Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Number of Videos
          </label>
          <input
            type="number"
            min="1"
            max="50"
            value={numVideos}
            onChange={(e) => setNumVideos(Number(e.target.value))}
            className="input"
            disabled={isGenerating}
          />
          <p className="text-xs text-gray-500 mt-1">
            Maximum 50 videos per batch
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Caption Style
          </label>
          <select
            value={captionStyle}
            onChange={(e) => setCaptionStyle(e.target.value)}
            className="select"
            disabled={isGenerating}
          >
            {captionStyles.map((style) => (
              <option key={style.value} value={style.value}>
                {style.label} {style.recommended ? '⭐' : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Estimation */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-medium text-blue-800">Batch Estimation</h4>
            <p className="text-sm text-blue-600">
              {numVideos} videos • Estimated time: ~{Math.ceil(estimatedTime / 60)} minutes
            </p>
          </div>
          <Clock className="w-5 h-5 text-blue-600" />
        </div>
      </div>

      {/* Warning for large batches */}
      {numVideos > 20 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
            <div>
              <h4 className="font-medium text-yellow-800">Large Batch Warning</h4>
              <p className="text-sm text-yellow-700">
                Generating {numVideos} videos will take significant time and resources. 
                Consider starting with a smaller batch.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={generateBatch}
        disabled={isGenerating || numVideos < 1}
        className="btn-primary w-full flex items-center justify-center"
      >
        {isGenerating ? (
          <>
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
            Generating Batch...
          </>
        ) : (
          <>
            <Layers className="w-4 h-4 mr-2" />
            Generate {numVideos} Videos
          </>
        )}
      </button>

      {/* Progress Display */}
      {currentJob && (
        <div className="bg-gray-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium">
              {currentJob.status === 'completed' ? '✅ Batch Complete' : 
               currentJob.status === 'failed' ? '❌ Batch Failed' : 
               '📊 Batch Processing'}
            </h3>
            {currentJob.processing_time && (
              <span className="text-sm text-gray-500 flex items-center">
                <Clock className="w-4 h-4 mr-1" />
                {Math.round(currentJob.processing_time / 60)}m
              </span>
            )}
          </div>

          {/* Progress Bar */}
          {currentJob.status === 'processing' && (
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>{currentJob.message}</span>
                <span>{Math.round(currentJob.progress)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="progress-bar h-2 rounded-full transition-all duration-300"
                  style={{ width: `${currentJob.progress}%` }}
                ></div>
              </div>
            </div>
          )}

          {/* Result */}
          {currentJob.status === 'completed' && currentJob.result && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-medium text-green-800 mb-2">Batch Completed Successfully!</h4>
              <div className="text-sm text-green-700 space-y-1">
                <p>• Generated {currentJob.result.total_videos} videos</p>
                <p>• Caption style: {currentJob.result.caption_style}</p>
                <p>• Output directory: {currentJob.result.output_directory}</p>
              </div>
            </div>
          )}

          {/* Error */}
          {currentJob.status === 'failed' && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800">{currentJob.message}</p>
            </div>
          )}
        </div>
      )}

      {/* Batch Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-medium text-gray-800 mb-2">📊 Batch Generation Features</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Cycles through all available scripts</li>
          <li>• Creates multiple variations for uniqueness</li>
          <li>• Uses cached captions for faster processing</li>
          <li>• Automatic uniqueness tracking</li>
          <li>• Progress monitoring and error handling</li>
        </ul>
      </div>
    </div>
  )
}