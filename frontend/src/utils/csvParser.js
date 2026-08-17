import Papa from 'papaparse';

/**
 * Safely parses a JSON string or returns fallback.
 */
function safeJsonParse(val, fallback = []) {
  if (!val) return fallback;
  if (typeof val !== 'string') {
    return Array.isArray(val) ? val : fallback;
  }
  try {
    const parsed = JSON.parse(val);
    return Array.isArray(parsed) ? parsed : fallback;
  } catch (err) {
    console.warn('JSON parse fallback for value:', val, err);
    return fallback;
  }
}

/**
 * Derives risk category from risk_score if missing or invalid.
 */
export function deriveRiskCategory(category, score) {
  if (category && ['High', 'Medium', 'Low'].includes(String(category).trim())) {
    return String(category).trim();
  }
  const numericScore = parseFloat(score) || 0;
  if (numericScore >= 70) return 'High';
  if (numericScore >= 40) return 'Medium';
  return 'Low';
}

/**
 * Standardizes a single parsed patient row.
 */
export function sanitizePatientRecord(row) {
  const score = parseFloat(row.risk_score) || (parseFloat(row.risk_probability) * 100) || 0;
  const category = deriveRiskCategory(row.risk_category, score);
  
  // Safety guardrail flag normalization
  let safetyFlag = false;
  if (row.safety_guardrail_flag === true || row.safety_guardrail_flag === 1 || String(row.safety_guardrail_flag).toLowerCase() === 'true') {
    safetyFlag = true;
  }

  // Parse JSON-encoded complex columns
  const topPos = safeJsonParse(row.top_positive_factors, []);
  const topNeg = safeJsonParse(row.top_negative_factors, []);
  const navOpps = safeJsonParse(row.navigation_opportunities, []);

  return {
    ...row,
    member_id: String(row.member_id || `M_${Math.random().toString(36).substr(2, 6)}`),
    risk_probability: parseFloat(row.risk_probability) || (score / 100),
    risk_score: score,
    risk_category: category,
    predicted_frequent_ED: parseInt(row.predicted_frequent_ED, 10) || (score >= 40 ? 1 : 0),
    safety_guardrail_flag: safetyFlag,
    safety_guardrail_message: row.safety_guardrail_message && row.safety_guardrail_message !== 'nan' && row.safety_guardrail_message !== 'null'
      ? String(row.safety_guardrail_message)
      : null,
    top_positive_factors: topPos,
    top_negative_factors: topNeg,
    navigation_opportunities: navOpps,
    // Demographics and chronic flags fallbacks
    age: parseInt(row.age, 10) || null,
    gender: row.gender || (row.gender_M === 1 ? 'M' : row.gender_M === 0 ? 'F' : 'N/A'),
    diabetes: parseInt(row.diabetes, 10) || 0,
    copd: parseInt(row.copd, 10) || 0,
    hypertension: parseInt(row.hypertension, 10) || 0,
    chf: parseInt(row.chf, 10) || 0,
    asthma: parseInt(row.asthma, 10) || 0,
    ckd: parseInt(row.ckd, 10) || 0,
    num_chronic_conditions: parseInt(row.num_chronic_conditions, 10) || 0,
    transportation_barrier: parseInt(row.transportation_barrier, 10) || 0,
    telehealth_available: parseInt(row.telehealth_available, 10) || 0,
    pcp_distance_miles: parseFloat(row.pcp_distance_miles) || null,
    urgent_care_distance_miles: parseFloat(row.urgent_care_distance_miles) || null,
    clinical_burden: parseFloat(row.clinical_burden) || null,
    access_burden: parseFloat(row.access_burden) || null,
  };
}

/**
 * Parses CSV file using PapaParse.
 */
export function parseScoredPatientsCSV(fileOrString) {
  return new Promise((resolve, reject) => {
    Papa.parse(fileOrString, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (!results.data || results.data.length === 0) {
          reject(new Error('The uploaded CSV file is empty.'));
          return;
        }
        const cleanedData = results.data
          .filter(row => row && row.member_id)
          .map(sanitizePatientRecord);

        if (cleanedData.length === 0) {
          reject(new Error('No valid patient records found in CSV file. Check column names.'));
          return;
        }

        resolve(cleanedData);
      },
      error: (err) => {
        reject(err);
      }
    });
  });
}
