/**
 * Utility functions for population aggregation, KPI calculations, and chart data formatting.
 */

// Colors matching the dark mode glassmorphism clinical palette
export const RISK_COLORS = {
  High: '#ef4444',    // Bright Red / Coral
  Medium: '#f59e0b',  // Amber / Warm Gold
  Low: '#10b981',     // Emerald Green
};

export const CONDITION_LABELS = {
  diabetes: 'Diabetes',
  copd: 'COPD',
  hypertension: 'Hypertension',
  chf: 'CHF',
  asthma: 'Asthma',
  ckd: 'CKD',
};

/**
 * Computes Population Overview KPIs.
 */
export function computePopulationKpis(patients = []) {
  const total = patients.length;
  if (total === 0) {
    return {
      totalPatients: 0,
      highRiskCount: 0, highRiskPct: 0,
      medRiskCount: 0, medRiskPct: 0,
      lowRiskCount: 0, lowRiskPct: 0,
      safetyFlagCount: 0, safetyFlagPct: 0,
      activeNavOppCount: 0, activeNavOppPct: 0,
    };
  }

  let high = 0, med = 0, low = 0, safety = 0, navOppsCount = 0;

  patients.forEach(p => {
    if (p.risk_category === 'High') high++;
    else if (p.risk_category === 'Medium') med++;
    else low++;

    if (p.safety_guardrail_flag) safety++;

    if (Array.isArray(p.navigation_opportunities) && p.navigation_opportunities.length > 0) {
      navOppsCount++;
    }
  });

  return {
    totalPatients: total,
    highRiskCount: high,
    highRiskPct: ((high / total) * 100).toFixed(1),
    medRiskCount: med,
    medRiskPct: ((med / total) * 100).toFixed(1),
    lowRiskCount: low,
    lowRiskPct: ((low / total) * 100).toFixed(1),
    safetyFlagCount: safety,
    safetyFlagPct: ((safety / total) * 100).toFixed(1),
    activeNavOppCount: navOppsCount,
    activeNavOppPct: ((navOppsCount / total) * 100).toFixed(1),
  };
}

/**
 * Formats Donut Chart data for Risk Category Distribution.
 */
export function getRiskDistributionChartData(patients = []) {
  const kpis = computePopulationKpis(patients);
  return [
    { name: 'High Risk', value: kpis.highRiskCount, color: RISK_COLORS.High, pct: kpis.highRiskPct },
    { name: 'Medium Risk', value: kpis.medRiskCount, color: RISK_COLORS.Medium, pct: kpis.medRiskPct },
    { name: 'Low Risk', value: kpis.lowRiskCount, color: RISK_COLORS.Low, pct: kpis.lowRiskPct },
  ].filter(item => item.value > 0 || patients.length === 0);
}

/**
 * Formats Stacked Bar Chart data: Chronic Condition Prevalence by Risk Category.
 * User decision: "Use Stacked Bar charts for Chronic Burden vs Risk Categories"
 */
export function getChronicPrevalenceStackedChartData(patients = []) {
  const conditions = ['diabetes', 'copd', 'hypertension', 'chf', 'asthma', 'ckd'];
  
  return conditions.map(condKey => {
    const counts = { High: 0, Medium: 0, Low: 0 };
    patients.forEach(p => {
      if (p[condKey] === 1) {
        const cat = p.risk_category || 'Low';
        if (counts[cat] !== undefined) {
          counts[cat]++;
        }
      }
    });

    return {
      condition: CONDITION_LABELS[condKey] || condKey,
      High: counts.High,
      Medium: counts.Medium,
      Low: counts.Low,
      Total: counts.High + counts.Medium + counts.Low,
    };
  });
}

/**
 * Formats Navigation Opportunity Type breakdown.
 */
export function getNavigationOpportunityBreakdown(patients = []) {
  const counts = {
    PCP: { name: 'PCP Navigation', key: 'PCP', count: 0, color: '#38bdf8', icon: 'UserCheck' },
    Transportation: { name: 'Transportation Support', key: 'Transportation', count: 0, color: '#a855f7', icon: 'Car' },
    Telehealth: { name: 'Telehealth Access', key: 'Telehealth', count: 0, color: '#34d399', icon: 'Video' },
    UrgentCare: { name: 'Urgent Care Navigation', key: 'UrgentCare', count: 0, color: '#f43f5e', icon: 'Building2' },
  };

  const eligiblePatients = patients.filter(
    p => Array.isArray(p.navigation_opportunities) && p.navigation_opportunities.length > 0 && !p.safety_guardrail_flag
  );

  eligiblePatients.forEach(p => {
    p.navigation_opportunities.forEach(opp => {
      const oppStr = String(opp).toLowerCase();
      if (oppStr.includes('pcp')) counts.PCP.count++;
      else if (oppStr.includes('transport')) counts.Transportation.count++;
      else if (oppStr.includes('telehealth')) counts.Telehealth.count++;
      else if (oppStr.includes('urgent')) counts.UrgentCare.count++;
    });
  });

  return Object.values(counts);
}

/**
 * Formats Risk Score Histogram Bins (0-10, 10-20, ... 90-100).
 */
export function getRiskScoreHistogram(patients = []) {
  const bins = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}-${(i + 1) * 10}`,
    min: i * 10,
    max: (i + 1) * 10,
    count: 0,
  }));

  patients.forEach(p => {
    const score = p.risk_score || 0;
    const binIdx = Math.min(Math.floor(score / 10), 9);
    bins[binIdx].count++;
  });

  return bins;
}

/**
 * Formats Average Risk Score grouped by Chronic Condition Count (0, 1, 2, 3, 4+).
 */
export function getRiskByChronicCount(patients = []) {
  const groups = {
    '0 Conditions': { totalScore: 0, count: 0 },
    '1 Condition': { totalScore: 0, count: 0 },
    '2 Conditions': { totalScore: 0, count: 0 },
    '3 Conditions': { totalScore: 0, count: 0 },
    '4+ Conditions': { totalScore: 0, count: 0 },
  };

  patients.forEach(p => {
    const num = p.num_chronic_conditions || 0;
    let label = '0 Conditions';
    if (num === 1) label = '1 Condition';
    else if (num === 2) label = '2 Conditions';
    else if (num === 3) label = '3 Conditions';
    else if (num >= 4) label = '4+ Conditions';

    groups[label].totalScore += p.risk_score || 0;
    groups[label].count++;
  });

  return Object.entries(groups).map(([label, data]) => ({
    label,
    avgScore: data.count > 0 ? Number((data.totalScore / data.count).toFixed(2)) : 0,
    patientCount: data.count,
  }));
}

/**
 * Formats Average Risk Score for Transportation Barrier = 1 vs 0.
 */
export function getAccessBarrierImpact(patients = []) {
  let withBarrierScore = 0, withBarrierCount = 0;
  let withoutBarrierScore = 0, withoutBarrierCount = 0;

  patients.forEach(p => {
    if (p.transportation_barrier === 1) {
      withBarrierScore += p.risk_score || 0;
      withBarrierCount++;
    } else {
      withoutBarrierScore += p.risk_score || 0;
      withoutBarrierCount++;
    }
  });

  return [
    {
      group: 'Transportation Barrier Present',
      avgScore: withBarrierCount > 0 ? Number((withBarrierScore / withBarrierCount).toFixed(2)) : 0,
      count: withBarrierCount,
    },
    {
      group: 'No Transportation Barrier',
      avgScore: withoutBarrierCount > 0 ? Number((withoutBarrierScore / withoutBarrierCount).toFixed(2)) : 0,
      count: withoutBarrierCount,
    },
  ];
}

/**
 * Computes summary statistics table for Trends & Reports tab.
 */
export function getTrendsSummaryStats(patients = []) {
  if (patients.length === 0) return null;

  const scores = patients.map(p => p.risk_score || 0).sort((a, b) => a - b);
  const meanScore = scores.reduce((sum, s) => sum + s, 0) / scores.length;
  
  const mid = Math.floor(scores.length / 2);
  const medianScore = scores.length % 2 !== 0 ? scores[mid] : (scores[mid - 1] + scores[mid]) / 2;

  const highRiskGroup = patients.filter(p => p.risk_category === 'High');
  const lowRiskGroup = patients.filter(p => p.risk_category === 'Low');

  const validHighAge = highRiskGroup.map(p => p.age).filter(a => typeof a === 'number' && !isNaN(a));
  const validLowAge = lowRiskGroup.map(p => p.age).filter(a => typeof a === 'number' && !isNaN(a));

  const meanHighAge = validHighAge.length > 0 ? validHighAge.reduce((sum, a) => sum + a, 0) / validHighAge.length : null;
  const meanLowAge = validLowAge.length > 0 ? validLowAge.reduce((sum, a) => sum + a, 0) / validLowAge.length : null;

  return {
    meanRiskScore: meanScore.toFixed(2),
    medianRiskScore: medianScore.toFixed(2),
    minRiskScore: scores[0].toFixed(2),
    maxRiskScore: scores[scores.length - 1].toFixed(2),
    highRiskPatientCount: highRiskGroup.length,
    lowRiskPatientCount: lowRiskGroup.length,
    meanHighRiskAge: meanHighAge !== null ? meanHighAge.toFixed(1) : 'N/A',
    meanLowRiskAge: meanLowAge !== null ? meanLowAge.toFixed(1) : 'N/A',
  };
}
