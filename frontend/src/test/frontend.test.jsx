import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Navbar from '../components/Navbar';
import SessionStats from '../components/SessionStats';
import QualityScoreCard from '../components/QualityScoreCard';
import { ProcessingStatusBadge, DecisionBadge } from '../components/StatusBadge';

describe('Face Quality Lab UI Components', () => {
  it('renders Navbar with actual device and healthy model indicators', () => {
    const health = {
      status: 'ok',
      device: 'cpu',
      cuda_available: false,
      model_loaded: true,
      model_variant: 's',
      supported_detectors: ['scrfd', 'retinaface'],
      timestamp: '2026-10-18T00:00:00Z',
    };

    render(
      <Navbar
        activeTab="single"
        setActiveTab={() => {}}
        health={health}
        loadingHealth={false}
      />
    );

    expect(screen.getByText('Face Quality Lab')).toBeInTheDocument();
    expect(screen.getByText('CPU')).toBeInTheDocument();
    expect(screen.getByText('Model Ready')).toBeInTheDocument();
  });

  it('renders Navbar in connecting / loading state', () => {
    render(
      <Navbar
        activeTab="single"
        setActiveTab={() => {}}
        health={null}
        loadingHealth={true}
      />
    );

    expect(screen.getByText('Connecting...')).toBeInTheDocument();
  });

  it('renders SessionStats with honest backend metrics', () => {
    const stats = {
      session_start_time: '2026-10-18T10:00:00Z',
      total_processed: 42,
      faces_detected: 40,
      detection_failures: 2,
      quality_decisions_completed: 0,
      not_evaluated_count: 40,
      current_config: {
        primary_detector: 'scrfd',
        fallback_detector: 'retinaface',
        scrfd_input_size: 640,
        head_pose: false,
      },
    };

    render(<SessionStats stats={stats} />);

    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('40')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText(/40 NOT_EVALUATED/i)).toBeInTheDocument();
  });

  it('renders QualityScoreCard with honest score and uncalibrated threshold notice', () => {
    render(
      <QualityScoreCard
        score={0.8593}
        decision="NOT_EVALUATED"
        decisionReason="no calibrated score thresholds configured"
        status="OK"
      />
    );

    // Exact score formatting
    expect(screen.getByText('0.8593')).toBeInTheDocument();
    expect(screen.getByText('NOT_EVALUATED')).toBeInTheDocument();
    // Mandatory threshold notice
    expect(
      screen.getByText('Score available — decision threshold not calibrated.')
    ).toBeInTheDocument();
    // Verify NOT labeled as ACCEPT or REJECT
    expect(screen.queryByText('ACCEPT')).not.toBeInTheDocument();
    expect(screen.queryByText('REJECT')).not.toBeInTheDocument();
  });

  it('renders QualityScoreCard empty/failure state when no face is detected', () => {
    render(
      <QualityScoreCard
        score={null}
        decision="NOT_EVALUATED"
        decisionReason="no quality score"
        status="NO_FACE"
      />
    );

    expect(screen.getByText('No Quality Score Available')).toBeInTheDocument();
    expect(
      screen.getByText(/No usable face was detected to feed the 112×112 quality backbone/i)
    ).toBeInTheDocument();
    expect(screen.queryByText('Blurry')).not.toBeInTheDocument();
  });

  it('renders StatusBadges correctly for OK, NO_FACE, and READ_ERROR', () => {
    const { rerender } = render(<ProcessingStatusBadge status="OK" />);
    expect(screen.getByText('OK')).toBeInTheDocument();

    rerender(<ProcessingStatusBadge status="NO_FACE" />);
    expect(screen.getByText('NO_FACE')).toBeInTheDocument();
    // Must never render "Blurry" for NO_FACE
    expect(screen.queryByText('Blurry')).not.toBeInTheDocument();

    rerender(<ProcessingStatusBadge status="READ_ERROR" />);
    expect(screen.getByText('READ_ERROR')).toBeInTheDocument();
  });

  it('renders DecisionBadge correctly for NOT_EVALUATED', () => {
    render(<DecisionBadge decision="NOT_EVALUATED" />);
    expect(screen.getByText('NOT_EVALUATED')).toBeInTheDocument();
  });
});
