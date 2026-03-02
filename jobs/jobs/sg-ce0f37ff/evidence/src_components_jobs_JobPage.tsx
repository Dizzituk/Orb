// src/components/jobs/JobPage.tsx
/**
 * Placeholder job page — shows icon, name, and "coming soon" message.
 * Debug will get a real implementation in a separate job.
 */

import type { JobType } from '@/types';
import { InvestmentsView } from '../investments/InvestmentsView';
import { DebugView } from '../debug/DebugView';
import { FinanceView } from '../finance/FinanceView';
import { ContentView } from '../content/ContentView';
import { LifestyleView } from '../lifestyle/LifestyleView';
import { SocialMediaView } from '../social-media/SocialMediaView';
import { BuildsView } from '../builds/BuildsView';

interface JobPageProps {
  jobType: JobType;
  onBackToChat: () => void;
}

const JOB_META: Record<JobType, { icon: string; label: string; description: string }> = {
  health_fitness: {
    icon: '💪',
    label: 'Health & Fitness',
    description: 'Workout programming, nutrition tracking, progress analytics, and personalised fitness plans.',
  },
  investments: {
    icon: '📈',
    label: 'Investments',
    description: 'Portfolio dashboard, stock/crypto tracking, ASTRA-curated news feed for your holdings.',
  },
  accounts: {
    icon: '📊',
    label: 'Accounts',
    description: 'Bank statement OCR, QuickBooks sync, categorised spending, Making Tax Digital compliance.',
  },
  content: {
    icon: '🎬',
    label: 'Content',
    description: 'Content pipeline dashboard — daily review, approval workflow, publishing queue, analytics.',
  },
  social_media: {
    icon: '📱',
    label: 'Social Media',
    description: 'Scheduling, analytics, cross-platform posting, engagement tracking, and audience insights.',
  },
  website: {
    icon: '🌐',
    label: 'Website',
    description: 'Client-facing website builder, CMS management, SEO tools, and deployment pipeline.',
  },
  education: {
    icon: '📚',
    label: 'Education',
    description: 'Curated learning paths, course tracking, skill development, and knowledge management.',
  },
  debug: {
    icon: '🔧',
    label: 'Debug',
    description: 'Conversational debug agent with full codebase context, log analysis, and pipeline diagnostics.',
  },
  project_builds: {
    icon: '🏗️',
    label: 'Project Builds',
    description: 'ASTRA pipeline workspace — build apps from natural language through Weaver, SpecGate, and Implementer.',
  },
};

export function JobPage({ jobType, onBackToChat }: JobPageProps) {
  // Lifestyle has a full dashboard
  if (jobType === 'health_fitness') {
    return <LifestyleView />;
  }

  // Investments has a full dashboard — render it directly
  if (jobType === 'investments') {
    return <InvestmentsView />;
  }

  // Accounts has the full finance dashboard
  if (jobType === 'accounts') {
    return <FinanceView />;
  }

  // Debug has its own conversational assistant view
  if (jobType === 'debug') {
    return <DebugView />;
  }

  // Content Hub — project-based content creation pipeline
  if (jobType === 'content') {
    return <ContentView />;
  }

  // Social Media — unified dashboard, scheduling, engagement
  if (jobType === 'social_media') {
    return <SocialMediaView />;
  }

  // Project Builds — pipeline workspace
  if (jobType === 'project_builds') {
    return <BuildsView />;
  }

  const meta = JOB_META[jobType];

  return (
    <div className="job-page">
      <div className="job-page-content">
        <span className="job-page-icon">{meta.icon}</span>
        <h1 className="job-page-title">{meta.label}</h1>
        <p className="job-page-description">{meta.description}</p>
        <div className="job-page-badge">Coming Soon</div>
        <button className="job-page-back" onClick={onBackToChat}>
          ← Back to Chat
        </button>
      </div>
    </div>
  );
}

