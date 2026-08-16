import React from 'react';
import { UserCheck, Car, Video, Building2, CheckCircle } from 'lucide-react';

export default function NavigationOpportunityCard({ opportunityText }) {
  const text = String(opportunityText || '');
  const textLower = text.toLowerCase();

  let Icon = CheckCircle;
  let colorStyle = 'border-sky-500/30 bg-sky-950/20 text-sky-300';
  let badgeColor = 'bg-sky-500/20 text-sky-300 border-sky-500/30';
  let title = 'Care Navigation Opportunity';

  if (textLower.includes('pcp')) {
    Icon = UserCheck;
    colorStyle = 'border-sky-500/30 bg-sky-950/20 text-sky-200';
    badgeColor = 'bg-sky-500/20 text-sky-300 border-sky-500/30';
    title = 'Primary Care Outpatient Navigation';
  } else if (textLower.includes('transport')) {
    Icon = Car;
    colorStyle = 'border-purple-500/30 bg-purple-950/20 text-purple-200';
    badgeColor = 'bg-purple-500/20 text-purple-300 border-purple-500/30';
    title = 'Transportation Support Resource';
  } else if (textLower.includes('telehealth')) {
    Icon = Video;
    colorStyle = 'border-emerald-500/30 bg-emerald-950/20 text-emerald-200';
    badgeColor = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
    title = 'Telehealth Virtual Care Access';
  } else if (textLower.includes('urgent')) {
    Icon = Building2;
    colorStyle = 'border-amber-500/30 bg-amber-950/20 text-amber-200';
    badgeColor = 'bg-amber-500/20 text-amber-300 border-amber-500/30';
    title = 'Urgent Care Alternative';
  }

  return (
    <div className={`p-4 rounded-xl border backdrop-blur-md transition-all duration-300 hover:scale-[1.01] ${colorStyle}`}>
      <div className="flex items-start gap-3">
        <div className={`p-2.5 rounded-lg border shrink-0 ${badgeColor}`}>
          <Icon className="w-5 h-5" />
        </div>
        <div className="space-y-1">
          <div className="text-xs font-bold uppercase tracking-wider opacity-80">
            {title}
          </div>
          <p className="text-sm font-medium leading-snug">
            {text}
          </p>
        </div>
      </div>
    </div>
  );
}
