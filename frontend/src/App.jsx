import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import SessionStats from './components/SessionStats';
import SingleImageAnalysis from './components/SingleImageAnalysis';
import BatchEvaluation from './components/BatchEvaluation';
import SettingsView from './components/SettingsView';
import { fetchHealth, fetchConfig, fetchSessionStats } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState(null);
  const [sessionStats, setSessionStats] = useState(null);
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [loadingConfig, setLoadingConfig] = useState(true);

  // Fetch system state from backend
  const loadSystemState = useCallback(async () => {
    try {
      setLoadingHealth(true);
      const [healthData, configData, statsData] = await Promise.all([
        fetchHealth().catch(() => ({ status: 'error', device: 'offline', cuda_available: false })),
        fetchConfig().catch(() => null),
        fetchSessionStats().catch(() => null),
      ]);
      setHealth(healthData);
      setConfig(configData);
      setSessionStats(statsData);
    } finally {
      setLoadingHealth(false);
      setLoadingConfig(false);
    }
  }, []);

  const refreshSessionStats = useCallback(async () => {
    try {
      const stats = await fetchSessionStats();
      setSessionStats(stats);
    } catch {
      // Ignore background refresh errors
    }
  }, []);

  useEffect(() => {
    loadSystemState();
  }, [loadSystemState]);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        loadingHealth={loadingHealth}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Session Stats Banner */}
        <SessionStats stats={sessionStats} onRefresh={refreshSessionStats} />

        {/* Tab Content */}
        {activeTab === 'single' && (
          <SingleImageAnalysis onAnalysisComplete={refreshSessionStats} />
        )}

        {activeTab === 'batch' && (
          <BatchEvaluation onBatchComplete={refreshSessionStats} />
        )}

        {activeTab === 'settings' && (
          <SettingsView
            config={config}
            health={health}
            loading={loadingConfig}
            onConfigUpdated={loadSystemState}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-slate-700">Face Quality Lab</span>
            <span>•</span>
            <span>eDifFIQA PyTorch Face Image Quality Assessment</span>
          </div>
          <div className="flex items-center space-x-3 text-slate-400">
            <span>SCRFD-10G / RetinaFace</span>
            <span>•</span>
            <span>UniFace HeadPose</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
