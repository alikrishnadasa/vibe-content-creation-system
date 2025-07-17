'use client'

import { useState, useEffect } from 'react'
import { Play, Zap, Clock, CheckCircle } from 'lucide-react'

interface Script {
  name: string
  path: string
  has_cached_captions: boolean
  cache_path?: string
}

interface GenerationJob {
  job_id: string
  status: string
  progress: number
  message: string
  processing_time?: number
  result?: {
    output_path: string
    processing_time: number
    target_achieved: boolean
    file_size: number
  }
}

export default function VideoGenerator() {
  const [scripts, setScripts] = useState<Script[]>([])
  const [selectedScript, setSelectedScript] = useState('')
  const [captionStyle, setCaptionStyle] = useState('default')
  const [variationNumber, setVariationNumber] = useState(1)
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentJob, setCurrentJob] = useState<GenerationJob | null>(null)
  const [loading, setLoading] = useState(true)

  const captionStyles = [
    { value: 'default', label: 'Default (Word-by-word)', recommended: true },
    { value: 'tiktok', label: 'TikTok Style' },
    { value: 'cinematic', label: 'Cinematic' },
    { value: 'minimal', label: 'Minimal' },
    { value: 'youtube', label: 'YouTube' },
    { value: 'karaoke', label: 'Karaoke' }
  ]

  useEffect(() => {
    fetchScripts()
  }, [])

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (currentJob && currentJob.status === 'processing') {
      interval = setInterval(() => {
        fetchJobStatus(currentJob.job_id)
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [currentJob])

  const fetchScripts = async () => {
    try {
      const response = await fetch('/api/scripts')
      const data = await response.json()
      setScripts(data.scripts)
      if (data.scripts.length > 0) {
        setSelectedScript(data.scripts[0].name)
      }
    } catch (error) {
      console.error('Failed to fetch scripts:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchJobStatus = async (jobId: string) => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`)
      const job = await response.json()
      setCurrentJob(job)
      
      if (job.status === 'completed' || job.status === 'failed') {
        setIsGenerating(false)
      }
    } catch (error) {
      console.error('Failed to fetch job status:', error)
    }
  }

  const generateVideo = async () => {
    if (!selectedScript) return

    setIsGenerating(true)
    setCurrentJob(null)

    try {
      const response = await fetch('/api/generate-video', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          script_name: selectedScript,
          caption_style: captionStyle,
          variation_number: variationNumber,
          music_sync: true,
          burn_in_captions: true
        }),
      })

      const data = await response.json()
      if (response.ok) {
        // Start polling for job status
        setCurrentJob({
          job_id: data.job_id,
          status: 'pending',
          progress: 0,
          message: 'Starting generation...'
        })
      } else {
        throw new Error(data.detail || 'Generation failed')
      }
    } catch (error) {
      console.error('Failed to start generation:', error)
      setIsGenerating(false)
    }
  }

  const downloadVideo = () => {
    if (currentJob?.result?.output_path) {
      window.open(`/api/download/${currentJob.job_id}`, '_blank')
    }
  }

  const selectedScriptData = scripts.find(s => s.name === selectedScript)
  const hasCache = selectedScriptData?.has_cached_captions || false

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        <span className="ml-2 text-gray-600">Loading scripts...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Configuration Form */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Audio Script
          </label>
          <select
            value={selectedScript}
            onChange={(e) => setSelectedScript(e.target.value)}
            className="select"
            disabled={isGenerating}
          >
            {scripts.map((script) => (
              <option key={script.name} value={script.name}>
                {script.name} {script.has_cached_captions ? '🚀' : ''}
              </option>
            ))}
          </select>
          {hasCache && (
            <p className="text-xs text-green-600 mt-1">
              🚀 Cached captions available - faster generation
            </p>
          )}
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

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Variation Number
          </label>
          <input
            type="number"
            min="1"
            max="10"
            value={variationNumber}
            onChange={(e) => setVariationNumber(Number(e.target.value))}
            className="input"
            disabled={isGenerating}
          />
          <p className="text-xs text-gray-500 mt-1">
            Different variations use different clip selections
          </p>
        </div>

        <div className="flex items-end">
          <button
            onClick={generateVideo}
            disabled={isGenerating || !selectedScript}
            className="btn-primary w-full flex items-center justify-center"
          >
            {isGenerating ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Generating...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Generate Video
              </>
            )}
          </button>
        </div>
      </div>

      {/* Progress Display */}
      {currentJob && (
        <div className="bg-gray-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium">
              {currentJob.status === 'completed' ? '✅ Generation Complete' : 
               currentJob.status === 'failed' ? '❌ Generation Failed' : 
               '⚡ Quantum Pipeline Processing'}
            </h3>
            {currentJob.processing_time && (
              <span className="text-sm text-gray-500 flex items-center">
                <Clock className="w-4 h-4 mr-1" />
                {currentJob.processing_time.toFixed(1)}s
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

          {/* Result Display */}
          {currentJob.status === 'completed' && currentJob.result && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="bg-white p-3 rounded-lg">
                  <div className="text-gray-500">Processing Time</div>
                  <div className="font-medium">
                    {currentJob.result.processing_time.toFixed(1)}s
                  </div>
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <div className="text-gray-500">Target Achieved</div>
                  <div className="font-medium">
                    {currentJob.result.target_achieved ? '✅ Yes' : '⏱️ No'}
                  </div>
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <div className="text-gray-500">File Size</div>
                  <div className="font-medium">
                    {(currentJob.result.file_size / (1024 * 1024)).toFixed(1)}MB
                  </div>
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <div className="text-gray-500">Quality</div>
                  <div className="font-medium">1080x1620</div>
                </div>
              </div>

              <button
                onClick={downloadVideo}
                className="btn bg-green-600 text-white hover:bg-green-700 w-full flex items-center justify-center"
              >
                <Play className="w-4 h-4 mr-2" />
                Download Video
              </button>
            </div>
          )}

          {/* Error Display */}
          {currentJob.status === 'failed' && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800">{currentJob.message}</p>
            </div>
          )}
        </div>
      )}

      {/* System Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-medium text-blue-800 mb-2">⚡ Quantum Pipeline Features</h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• Neural predictive caching for ultra-fast processing</li>
          <li>• Real MJAnime clips with intelligent content selection</li>
          <li>• Music synchronization with beat detection</li>
          <li>• Burned-in captions with multiple styles</li>
          <li>• 2:3 aspect ratio optimized for social media</li>
        </ul>
      </div>
    </div>
  )
}