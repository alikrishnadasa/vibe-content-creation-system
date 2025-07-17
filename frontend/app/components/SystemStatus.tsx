'use client'

import { useState, useEffect } from 'react'
import { Activity, Database, Zap, CheckCircle, AlertCircle } from 'lucide-react'

interface SystemHealth {
  status: string
  quantum_pipeline_ready: boolean
  available_scripts: number
  cached_caption_scripts: number
  target_resolution: [number, number]
  caption_styles: string[]
}

export default function SystemStatus() {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchHealth()
    const interval = setInterval(fetchHealth, 30000) // Update every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchHealth = async () => {
    try {
      const response = await fetch('/api/health')
      const data = await response.json()
      setHealth(data)
    } catch (error) {
      console.error('Failed to fetch health:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="card p-4">
        <div className="animate-pulse flex space-x-4">
          <div className="rounded-full bg-gray-300 h-10 w-10"></div>
          <div className="flex-1 space-y-2 py-1">
            <div className="h-4 bg-gray-300 rounded w-3/4"></div>
            <div className="h-4 bg-gray-300 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    )
  }

  if (!health) {
    return (
      <div className="card p-4 border-red-200 bg-red-50">
        <div className="flex items-center">
          <AlertCircle className="w-5 h-5 text-red-600 mr-2" />
          <span className="text-red-800">Unable to connect to backend</span>
        </div>
      </div>
    )
  }

  const isHealthy = health.status === 'healthy' && health.quantum_pipeline_ready

  return (
    <div className={`card p-4 ${isHealthy ? 'border-green-200 bg-green-50' : 'border-yellow-200 bg-yellow-50'}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center">
            {isHealthy ? (
              <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
            ) : (
              <AlertCircle className="w-5 h-5 text-yellow-600 mr-2" />
            )}
            <span className={`font-medium ${isHealthy ? 'text-green-800' : 'text-yellow-800'}`}>
              System Status: {isHealthy ? 'Ready' : 'Initializing'}
            </span>
          </div>

          <div className="hidden md:flex items-center space-x-6 text-sm">
            <div className="flex items-center">
              <Zap className="w-4 h-4 text-blue-600 mr-1" />
              <span className="text-gray-700">
                Quantum Pipeline: {health.quantum_pipeline_ready ? 'Active' : 'Loading'}
              </span>
            </div>

            <div className="flex items-center">
              <Database className="w-4 h-4 text-purple-600 mr-1" />
              <span className="text-gray-700">
                Scripts: {health.available_scripts}
              </span>
            </div>

            <div className="flex items-center">
              <Activity className="w-4 h-4 text-green-600 mr-1" />
              <span className="text-gray-700">
                Cached: {health.cached_caption_scripts}
              </span>
            </div>
          </div>
        </div>

        <div className="text-xs text-gray-500">
          Resolution: {health.target_resolution[0]}×{health.target_resolution[1]}
        </div>
      </div>

      {/* Mobile view */}
      <div className="md:hidden mt-3 grid grid-cols-3 gap-2 text-xs">
        <div className="text-center">
          <div className="text-gray-500">Pipeline</div>
          <div className="font-medium">{health.quantum_pipeline_ready ? 'Active' : 'Loading'}</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Scripts</div>
          <div className="font-medium">{health.available_scripts}</div>
        </div>
        <div className="text-center">
          <div className="text-gray-500">Cached</div>
          <div className="font-medium">{health.cached_caption_scripts}</div>
        </div>
      </div>
    </div>
  )
}