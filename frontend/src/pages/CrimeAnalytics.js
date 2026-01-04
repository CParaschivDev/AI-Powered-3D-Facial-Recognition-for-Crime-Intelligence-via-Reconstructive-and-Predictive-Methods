import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { buildCrimeContext, getCrimeMonthlyTrends, getCrimeLatestHotspots, getCrimeLsoaSeries, getCrimeSummary, getCrimeAggregatedTrends, listCrimeForces, listCrimeLsoas } from '../api';
import apiClient from '../api';
import {
  LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, BarChart, Bar
} from 'recharts';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import './CrimeAnalytics.css';

// Configure Leaflet default marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

function downloadBlob(filename, content, mime = 'text/plain') {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

const ALL_LSOAS_VALUE = '__ALL_LSOAS__';

async function exportChartPng(containerRef, filename) {
  if (!containerRef.current) return;
  const el = containerRef.current;
  const rect = el.getBoundingClientRect();
  const html2canvas = (await import('html2canvas')).default;
  const canvas = await html2canvas(el, { backgroundColor: '#fff', scale: 2, width: rect.width, height: rect.height, windowWidth: rect.width, windowHeight: rect.height });
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  });
}

const Section = ({ title, children }) => (
  <div className="analytics-section">
    <h3>{title}</h3>
    {children}
  </div>
);

export default function CrimeAnalytics() {
  const trendsRef = useRef(null);
  const hotspotsRef = useRef(null);
  const seriesRef = useRef(null);
  const [forceFilter, setForceFilter] = useState('');
  const [buildResult, setBuildResult] = useState(null);
  const [buildLoading, setBuildLoading] = useState(false);
  const [buildError, setBuildError] = useState(null);
  const [buildStatus, setBuildStatus] = useState(null);
  const [autoPoll, setAutoPoll] = useState(true);
  const [retrying, setRetrying] = useState(false);
  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [forceOptions, setForceOptions] = useState([]);
  const [lsoaOptions, setLsoaOptions] = useState([]);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await apiClient.get('/analytics/crime/context/status');
      const status = s.data?.status || null;
      setBuildStatus(status);
      if (status && !status.running) {
        setAutoPoll(false); // stop polling when finished
      }
    } catch (e) {
      // Leave buildError untouched unless we want to surface status failures
    }
  }, []);

  useEffect(() => {
    if (autoPoll) {
      const id = setInterval(() => {
        fetchStatus();
      }, 3000);
      return () => clearInterval(id);
    }
  }, [autoPoll, fetchStatus]);

  const [from, setFrom] = useState('2024-01');
  const [to, setTo] = useState('2025-07');
  const [selectedForces, setSelectedForces] = useState([]);
  const [trends, setTrends] = useState([]);
  const [trendsLoaded, setTrendsLoaded] = useState(false);
  const [trendsLoading, setTrendsLoading] = useState(false);
  const [trendsError, setTrendsError] = useState(null);

  const [hotForce, setHotForce] = useState('');
  const [hotspots, setHotspots] = useState([]);
  const [hotspotsLoaded, setHotspotsLoaded] = useState(false);
  const [hotLoading, setHotLoading] = useState(false);
  const [hotError, setHotError] = useState(null);

  const [lsoa, setLsoa] = useState(''); // Start with empty LSOA - will be populated by useEffect
  const [series, setSeries] = useState([]);
  const [seriesLoaded, setSeriesLoaded] = useState(false);
  const prevRunningRef = useRef(null);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const [seriesError, setSeriesError] = useState(null);

  const handleBuild = async () => {
    setBuildLoading(true); setBuildError(null); setBuildResult(null);
    try {
      const res = await buildCrimeContext(forceFilter || null);
      setBuildResult(res);
      // Immediately fetch status to show banner
      setAutoPoll(true);
      fetchStatus();
    } catch (e) {
      setBuildError(e?.response?.data?.detail || e.message);
    } finally {
      setBuildLoading(false);
    }
  };
  const handleCheckStatus = async () => { fetchStatus(); };

  const handleRetry = async () => {
    if (!buildStatus || buildStatus.running) return;
    setRetrying(true);
    try {
      await handleBuild();
    } finally {
      setRetrying(false);
    }
  };

  const handleLoadTrends = async () => {
    setTrendsLoading(true); setTrendsError(null);
    try {
      const data = await getCrimeMonthlyTrends(from, to);
      console.log('[CrimeAnalytics] trends response', { from, to, total: data.length });
      
      // Populate forceOptions if still empty
      if (!forceOptions.length && data.length) {
        const forcesAll = Array.from(new Set(data.map(r => r['Falls within']))).filter(Boolean).sort();
        setForceOptions(forcesAll);
      }
      
      // Apply force filter
      const filtered = selectedForces.length 
        ? data.filter(d => selectedForces.includes(d['Falls within'])) 
        : data;
      
      setTrends(filtered);
      setTrendsLoaded(true);
      
      if (filtered.length === 0 && !selectedForces.length) {
        console.warn('No trends data returned for range', { from, to });
        setTrendsError('No data available for the selected date range. Try adjusting the dates or clicking "Refresh All".');
      }
    } catch (e) {
      console.error('Failed to load trends:', e);
      setTrendsError(e?.response?.data?.detail || e.message || 'Failed to load trends data');
    } finally {
      setTrendsLoading(false);
    }
  };

  // Helpers for month arithmetic
  const parseMonth = (m) => { const [y, mo] = m.split('-').map(Number); return { y, mo }; };
  const formatMonth = (y, mo) => `${y}-${String(mo).padStart(2,'0')}`;
  const addMonths = (m, delta) => { const { y, mo } = parseMonth(m); let ny = y; let nmo = mo + delta; while (nmo > 12) { nmo -= 12; ny += 1; } while (nmo < 1) { nmo += 12; ny -= 1; } return formatMonth(ny, nmo); };

  const clampMonth = (m) => {
    if (!summary || !summary.min_month || !summary.max_month) return m;
    if (typeof summary.min_month !== 'string' || typeof summary.max_month !== 'string') return m;
    const minM = summary.min_month.slice(0,7);
    const maxM = summary.max_month.slice(0,7);
    if (m < minM) return minM;
    if (m > maxM) return maxM;
    return m;
  };

  const setFullRange = () => {
    // Defensive: ensure summary and month strings exist before slicing
    if (!summary || !summary.min_month || !summary.max_month) return;
    const min = summary.min_month && typeof summary.min_month === 'string' ? summary.min_month.slice(0,7) : null;
    const max = summary.max_month && typeof summary.max_month === 'string' ? summary.max_month.slice(0,7) : null;
    if (min) setFrom(min);
    if (max) setTo(max);
  };
  const setLastNMonths = (n) => {
    // Defensive: ensure we have a usable max_month string
    if (!summary || !summary.max_month || typeof summary.max_month !== 'string') return;
    const end = summary.max_month.slice(0,7);
    if (!end) return;
    const start = addMonths(end, - (n - 1));
    setFrom(start);
    setTo(end);
  };

  // Validate when from/to change: ensure from <= to and within dataset
  useEffect(() => {
    if (!summary || !summary.min_month) return;
    setFrom(f => clampMonth(f));
    setTo(t => clampMonth(t));
  }, [summary]);

  useEffect(() => {
    if (from > to) {
      // Auto-correct by setting to = from if user inverted
      setTo(from);
    }
  }, [from, to]);

  const handleLoadHotspots = async (force = hotForce) => {
    setHotLoading(true); setHotError(null);
    try {
      const data = await getCrimeLatestHotspots(force || undefined);
      console.log('[CrimeAnalytics] hotspots response', { force, total: data.length });
      setHotspots(data || []);
      setHotspotsLoaded(true);
      
      if (data.length === 0) {
        setHotError(`No hotspot data found${force ? ` for ${force}` : ''}. Try selecting a different force or click "Refresh All".`);
      }
    } catch (e) {
      console.error('Failed to load hotspots:', e);
      setHotError(e?.response?.data?.detail || e.message || 'Failed to load hotspots data');
    } finally {
      setHotLoading(false);
    }
  };

  const useSelectedForHotspots = () => {
    // If exactly one force selected, use that; if none, leave as-is; if multiple, show guidance
    if (selectedForces.length === 1) {
      setHotForce(selectedForces[0]);
    } else if (selectedForces.length === 0) {
      // use global (omit force)
      setHotForce('');
    } else {
      alert('Multiple forces selected. Please select a single force to view force-level hotspots, or clear selection to view global hotspots.');
    }
  };

  const handleLoadSeries = async () => {
    if (!hotForce && (!lsoa || lsoa.trim() === '' || lsoa === ALL_LSOAS_VALUE)) {
      setSeriesError('Please select a force in the hotspots section above, or choose a specific LSOA.');
      return;
    }

    if (lsoa === ALL_LSOAS_VALUE && !hotForce) {
      setSeriesError('Please select a force in the hotspots section above to view force-level trends.');
      return;
    }
    
    setSeriesLoading(true); setSeriesError(null);
    try {
      // FORCE-LEVEL VIEW: All LSOAs in {hotForce}
      if (lsoa === ALL_LSOAS_VALUE && hotForce) {
        const data = await getCrimeMonthlyTrends(from, to);
        console.log('[CrimeAnalytics] Loading force-level trends for', hotForce, 'from', from, 'to', to);

        const forceData = data.filter(r => r['Falls within'] === hotForce);

        if (!forceData.length) {
          setSeries([]);
          setSeriesLoaded(false);
          setSeriesError(`No trend data found for ${hotForce} in the selected date range.`);
          return;
        }

        // Aggregate by month and put everything under a single "All crimes" type
        const byMonth = forceData.reduce((acc, row) => {
          const month = row['Month'];
          const count = Number(row['crime_count'] || 0);
          if (!acc[month]) {
            acc[month] = {
              Month: month,
              'Crime type': 'All crimes',
              crime_count: 0
            };
          }
          acc[month].crime_count += count;
          return acc;
        }, {});

        const rawSeriesData = Object.values(byMonth);
        console.log('[CrimeAnalytics] Force-level raw series data:', rawSeriesData.length, 'records');

        setSeries(rawSeriesData);
        setSeriesLoaded(true);

        if (rawSeriesData.length === 0) {
          setSeriesError(`No trend data found for ${hotForce}. Try adjusting the date range.`);
        }

      // LSOA-LEVEL VIEW (unchanged)
      } else {
        const data = await getCrimeLsoaSeries(lsoa);
        console.log('[CrimeAnalytics] series response', { lsoa, total: data.length });
        setSeries(data || []);
        setSeriesLoaded(true);
        
        if (data.length === 0) {
          setSeriesError(`No time series data found for LSOA: ${lsoa}. Please check the LSOA name or select from the dropdown.`);
        }
      }
    } catch (e) {
      console.error('Failed to load series:', e);
      setSeriesError(e?.response?.data?.detail || e.message || 'Failed to load LSOA series data');
    } finally {
      setSeriesLoading(false);
    }
  };

  const handleLoadSummary = async () => {
    setSummaryLoading(true);
    try {
      const s = await getCrimeSummary();
      setSummary(s);
    } catch (e) { /* ignore for now */ }
    finally { setSummaryLoading(false); }
  };

  const refreshAll = async () => {
    await handleLoadSummary();
    await Promise.all([
      handleLoadTrends(),
      handleLoadHotspots(),
      handleLoadSeries(),
      (async () => {
        try {
          const agg = await getCrimeAggregatedTrends();
          // derive dropdown options
          const forces = Array.from(new Set(agg.map(r => r['Falls within']))).sort();
          const lsoas = Array.from(new Set(agg.map(r => r['LSOA name']).filter(Boolean))).sort();
          if (forces.length && !forceOptions.length) setForceOptions(forces);
          if (lsoas.length && !lsoaOptions.length) setLsoaOptions(lsoas);
        } catch {}
      })()
    ]);
  };

  // Initial summary load on mount
  useEffect(() => { handleLoadSummary(); }, []);

  // Prefetch distinct forces and LSOAs on mount
  useEffect(() => {
    (async () => {
      try {
        const forces = await listCrimeForces();
        if (forces.length) {
          setForceOptions(forces);
          console.log('[CrimeAnalytics] Loaded forces:', forces.length);
        }
        
        // Load global LSOAs since hotForce starts as ''
        try {
          const agg = await getCrimeAggregatedTrends();
          const lsoas = Array.from(new Set(agg.map(r => r['LSOA name']).filter(Boolean))).sort();
          if (lsoas.length) {
            setLsoaOptions(lsoas);
            setLsoa(''); // Start with no LSOA selected
            console.log('[CrimeAnalytics] Loaded global LSOAs:', lsoas.length);
          }
        } catch (e) {
          console.warn('Could not load global LSOAs:', e.message);
        }
      } catch (e) {
        console.warn('Could not prefetch force list:', e.message);
      }
    })();
  }, []);

  // After summary loads (and only once), auto-load default trends and hotspots
  useEffect(() => {
    if (summary && !trendsLoaded && !trendsLoading) {
      console.log('[CrimeAnalytics] Auto-loading trends with summary:', summary);
      handleLoadTrends();
    }
    if (summary && !hotspotsLoaded && !hotLoading) {
      console.log('[CrimeAnalytics] Auto-loading hotspots for:', hotForce);
      handleLoadHotspots();
    }
    // Don't auto-load series until user has selected an LSOA
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summary]);

  // Sync LSOAs when hotForce changes (either from filter or manual selection)
  useEffect(() => {
    let isMounted = true;
    
    const loadLsoas = async () => {
      if (!hotForce) {
        // If no force selected, load global LSOAs from aggregated trends
        try {
          console.log('[CrimeAnalytics] Loading global LSOAs from aggregated trends...');
          const agg = await getCrimeAggregatedTrends();
          const lsoas = Array.from(new Set(agg.map(r => r['LSOA name']).filter(Boolean))).sort();
          console.log('[CrimeAnalytics] Loaded global LSOAs:', lsoas.length, 'First 5:', lsoas.slice(0, 5));
          if (isMounted && lsoas.length > 0) {
            setLsoaOptions(lsoas);
            setLsoa(''); // Clear LSOA selection when switching to global
          }
        } catch (e) {
          console.error('[CrimeAnalytics] Failed to load global LSOAs from aggregated:', e);
          if (isMounted) {
            setLsoaOptions([]);
            setLsoa('');
          }
        }
      } else {
        // Load LSOAs for the specific force
        try {
          console.log('[CrimeAnalytics] Loading LSOAs for force:', hotForce);
          const forceLsoas = await listCrimeLsoas(hotForce);
          console.log('[CrimeAnalytics] Received LSOAs for', hotForce, ':', forceLsoas.length, 'First 5:', forceLsoas.slice(0, 5));
          
          if (isMounted) {
            if (forceLsoas.length > 0) {
              // CRITICAL: Update in correct order - options first, then value
              setLsoaOptions(forceLsoas);
              setSeries([]); // Clear series data when changing force
              setSeriesLoaded(false);
              // Use setTimeout to ensure options are set before value
              setTimeout(() => {
                if (isMounted) {
                  setLsoa(ALL_LSOAS_VALUE); // Default: view aggregated trends for the entire force
                  console.log('[CrimeAnalytics] Auto-selected force-level view for', hotForce);
                }
              }, 50);
            } else {
              // No LSOAs found for this force
              console.warn('[CrimeAnalytics] No LSOAs found for', hotForce);
              setLsoaOptions([]);
              setLsoa('');
            }
          }
        } catch (e) {
          console.error('[CrimeAnalytics] Failed to load LSOAs for force:', e);
          if (isMounted) {
            setLsoaOptions([]);
            setLsoa('');
          }
        }
      }
    };
    
    loadLsoas();
    
    return () => {
      isMounted = false;
    };
  }, [hotForce]);

  // Auto-load analytics after a successful build completion (transition running true -> false without error)
  useEffect(() => {
    const wasRunning = prevRunningRef.current;
    const isRunning = buildStatus?.running;
    if (wasRunning && wasRunning === true && isRunning === false && buildStatus && !buildStatus.error) {
      // Reset loaded flags and fetch fresh data
      setTrendsLoaded(false); setHotspotsLoaded(false); setSeriesLoaded(false);
      refreshAll();
    }
    prevRunningRef.current = isRunning;
  }, [buildStatus]);

  // Group trends by force for multi-series line chart
  const trendsByForce = trends.reduce((acc, row) => {
    const key = row['Falls within'];
    const date = row['Month'];
    const count = row['crime_count'];
    if (!acc[date]) acc[date] = { Month: date };
    acc[date][key] = (acc[date][key] || 0) + count;
    return acc;
  }, {});
  const trendsChartData = Object.values(trendsByForce).sort((a, b) => (a.Month > b.Month ? 1 : -1));
  const forceKeys = Array.from(new Set(trends.map(r => r['Falls within']))).slice(0, 8);

  // Series by crime type for a single LSOA
  const seriesByDate = series.reduce((acc, r) => {
    const date = r['Month'];
    if (!acc[date]) acc[date] = { Month: date };
    acc[date][r['Crime type']] = (acc[date][r['Crime type']] || 0) + r['crime_count'];
    return acc;
  }, {});
  const seriesChartData = Object.values(seriesByDate).sort((a, b) => (a.Month > b.Month ? 1 : -1));
  const crimeTypes = Array.from(new Set(series.map(r => r['Crime type']))).slice(0, 5);

  return (
    <div className="crime-analytics-container">
      <div className="crime-analytics-header">
        <h2>Crime Analytics</h2>
      </div>

      <div className="page-navigation">
        <a href="/dashboard">Dashboard</a>
        <a href="/upload">Upload</a>
        <a href="/crime-analytics" className="active">Crime Analytics</a>
      </div>

      <Section title="Dataset Summary & Controls">
        <div className="form-row">
          <button className="action-button" onClick={refreshAll}>Refresh All</button>
          <button className="action-button" onClick={handleLoadSummary} disabled={summaryLoading}>
            {summaryLoading ? (
              <div className="loading-indicator">
                <div className="spinner"></div>
                Loading Summary…
              </div>
            ) : 'Reload Summary'}
          </button>
          {summary && (
            <div className="summary-info">
              <strong>Rows:</strong> {summary.rows.toLocaleString()} &nbsp;|
              &nbsp;<strong>Month Range:</strong> {summary.min_month} → {summary.max_month} &nbsp;|
              &nbsp;<strong>Forces:</strong> {summary.forces} &nbsp;|
              &nbsp;<strong>LSOAs:</strong> {summary.lsoas}
            </div>
          )}
        </div>
        {summary && (
          <p className="hint-text">Enter date range within {summary.min_month?.slice(0,7)} and {summary.max_month?.slice(0,7)} for best results.</p>
        )}
        {summary && (!summary.min_month || !summary.max_month) && (
          <p className="warning-text">
            Dataset month range unavailable — some dataset metadata is missing. The range buttons are disabled until the summary has valid min/max months. Try <button className="action-button" onClick={handleLoadSummary}>Refresh Summary</button>.
          </p>
        )}
      </Section>

      <Section title="Build Crime Context (Parquet)">
        <div className="form-row">
          <input
            className="form-input"
            placeholder="Force filter (optional)"
            value={forceFilter}
            onChange={e => setForceFilter(e.target.value)}
          />
          <button className="action-button" onClick={handleBuild} disabled={buildLoading}>
            {buildLoading ? (
              <div className="loading-indicator">
                <div className="spinner"></div>
                Building…
              </div>
            ) : 'Build'}
          </button>
          <button className="action-button" onClick={handleCheckStatus}>Check Status</button>
          <label className="checkbox-label">
            <input
              className="checkbox-input"
              type="checkbox"
              checked={autoPoll}
              onChange={e => setAutoPoll(e.target.checked)}
            />
            Auto-refresh
          </label>
        </div>
        {buildError && <div className="error-message">{buildError}</div>}
        {buildResult && <pre style={{ maxHeight: 200, overflow: 'auto' }}>{JSON.stringify(buildResult, null, 2)}</pre>}
        {buildStatus && (
          <div className={`status-banner ${buildStatus.running ? 'running' : (buildStatus.error ? 'error' : 'completed')}`}>
            <div className="status-item"><strong>Status:</strong> {buildStatus.running ? 'Running' : (buildStatus.error ? 'Failed' : 'Completed')}</div>
            {buildStatus.started_at && <div className="status-item"><strong>Started:</strong> {buildStatus.started_at}</div>}
            {buildStatus.finished_at && <div className="status-item"><strong>Finished:</strong> {buildStatus.finished_at}</div>}
            {buildStatus.output_path && <div className="status-item"><strong>Output:</strong> {buildStatus.output_path}</div>}
            {buildStatus.size_bytes != null && <div className="status-item"><strong>Size:</strong> {buildStatus.size_bytes.toLocaleString()} bytes</div>}
            {buildStatus.files_total != null && <div className="status-item"><strong>Files:</strong> {buildStatus.files_processed}/{buildStatus.files_total}</div>}
            {buildStatus.rows_written != null && <div className="status-item"><strong>Rows Written:</strong> {buildStatus.rows_written.toLocaleString()}</div>}
            {buildStatus.last_file && buildStatus.running && <div className="status-item" style={{ maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}><strong>Current File:</strong> {buildStatus.last_file}</div>}
            {buildStatus.error && <div className="status-item" style={{ color: '#ff6b6b' }}><strong>Error:</strong> {buildStatus.error}</div>}
            {buildStatus.files_total > 0 && (
              <div className="progress-container">
                {(() => {
                  const pct = Math.min(100, Math.round((buildStatus.files_processed / buildStatus.files_total) * 100));
                  return (
                    <div>
                      <div className="progress-text">
                        <span>Progress</span>
                        <span>{pct}%</span>
                      </div>
                      <div className="progress-bar">
                        <div className={`progress-fill ${buildStatus.running ? 'running' : ''}`} style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
            {!buildStatus.running && buildStatus.error && (
              <div style={{ marginTop: 8 }}>
                <button className="action-button" disabled={retrying} onClick={handleRetry}>
                  {retrying ? (
                    <div className="loading-indicator">
                      <div className="spinner"></div>
                      Retrying...
                    </div>
                  ) : 'Retry Build'}
                </button>
              </div>
            )}
          </div>
        )}
      </Section>

      <Section title="Monthly Trends by Force">
        <div className="form-row">
          <input
            className="form-input"
            value={from}
            onChange={e => setFrom(e.target.value)}
            placeholder="YYYY-MM from"
          />
          <input
            className="form-input"
            value={to}
            onChange={e => setTo(e.target.value)}
            placeholder="YYYY-MM to"
          />
          <button className="action-button" onClick={handleLoadTrends} disabled={trendsLoading}>
            {trendsLoading ? (
              <div className="loading-indicator">
                <div className="spinner"></div>
                Loading…
              </div>
            ) : 'Load'}
          </button>
          {selectedForces.length > 0 && (
            <button className="action-button primary" onClick={async () => {
              await handleLoadTrends();
              if (selectedForces.length === 1) {
                const force = selectedForces[0];
                setHotForce(force); // This will trigger LSOA sync via useEffect
                await handleLoadHotspots(force);
              } else {
                setHotForce(''); // Global hotspots for multiple or no selection (triggers global LSOA sync)
                await handleLoadHotspots('');
              }
            }} disabled={trendsLoading || hotLoading}>
              {trendsLoading ? (
                <div className="loading-indicator">
                  <div className="spinner"></div>
                  Applying Filter…
                </div>
              ) : 'Apply Filter'}
            </button>
          )}
          <button
            className="export-button"
            title={trends.length === 0 ? 'Load data first' : ''}
            disabled={trends.length === 0}
            onClick={() => downloadBlob(`monthly_trends_${from}_${to}.csv`, toCsv(trends), 'text/csv')}
          >
            Export CSV
          </button>
          <button
            className="export-button"
            title={trends.length === 0 ? 'Load data first' : ''}
            disabled={trends.length === 0}
            onClick={() => exportChartPng(trendsRef, `monthly_trends_${from}_${to}.png`)}
          >
            Export PNG
          </button>
        </div>
        <div className="range-buttons">
          <button
            className="range-button"
            type="button"
            onClick={() => { setFullRange(); }}
            disabled={!(summary && summary.min_month && summary.max_month)}
          >
            Full Range
          </button>
          <button
            className="range-button"
            type="button"
            onClick={() => { setLastNMonths(12); }}
            disabled={!(summary && summary.max_month)}
          >
            Last 12M
          </button>
          <button
            className="range-button"
            type="button"
            onClick={() => { setLastNMonths(6); }}
            disabled={!(summary && summary.max_month)}
          >
            Last 6M
          </button>
          <button
            className="range-button"
            type="button"
            onClick={() => { setLastNMonths(3); }}
            disabled={!(summary && summary.max_month)}
          >
            Last 3M
          </button>
        </div>
        <div className="forces-selector">
          <label className="forces-label">Select Forces (optional – click to toggle; max 8 plotted):</label>
          <div className="forces-container">
            {forceOptions.map(f => {
              const active = selectedForces.includes(f);
              return (
                <button
                  key={f}
                  className={`force-button ${active ? 'active' : ''}`}
                  type="button"
                  onClick={() => setSelectedForces(prev => active ? prev.filter(x => x !== f) : [...prev, f])}
                >
                  {f}
                </button>
              );
            })}
          </div>
          <div className="hotspots-actions">
            <button
              className="hotspots-button"
              type="button"
              onClick={useSelectedForHotspots}
              disabled={forceOptions.length === 0}
            >
              Use Selected for Hotspots
            </button>
            <span className="hotspots-hint">Select one force and click this to set the hotspots view to that force; clear selection to see global hotspots.</span>
          </div>
          {selectedForces.length > 0 && <p className="selection-info">Selected {selectedForces.length} forces{selectedForces.length === 1 ? `: ${selectedForces[0]}` : ''}. Click "Apply Filter" to update the chart and hotspots view ({selectedForces.length === 1 ? 'filtered to selected force' : 'global view'}).</p>}
        </div>
        {trendsError && <div className="error-message">{trendsError}</div>}
        {!trendsLoading && trendsLoaded && trendsChartData.length === 0 && !trendsError && (
          <div className="no-data">No data for selected range.</div>
        )}
        {trendsChartData.length > 0 && (
          <div className="chart-container" ref={trendsRef}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendsChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="Month" />
                <YAxis />
                <Tooltip />
                <Legend />
                {forceKeys.map((k, idx) => (
                  <Line key={k} type="monotone" dataKey={k} stroke={["#8884d8","#82ca9d","#ffc658","#ff7300"][idx % 4]} dot={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      <Section title="Latest Hotspots (Top LSOAs) + Map">
        <div className="form-row">
          <select
            className="form-input"
            value={hotForce}
            onChange={e => setHotForce(e.target.value)}
          >
            <option value="">Global hotspots</option>
            {forceOptions.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
          <button className="action-button" onClick={handleLoadHotspots} disabled={hotLoading}>
            {hotLoading ? (
              <div className="loading-indicator">
                <div className="spinner"></div>
                Loading…
              </div>
            ) : 'Load'}
          </button>
          <button
            className="export-button"
            title={hotspots.length === 0 ? 'Load data first' : ''}
            disabled={hotspots.length === 0}
            onClick={() => downloadBlob(`hotspots_${hotForce}.csv`, toCsv(hotspots), 'text/csv')}
          >
            Export CSV
          </button>
          <button
            className="export-button"
            title={hotspots.length === 0 ? 'Load data first' : ''}
            disabled={hotspots.length === 0}
            onClick={() => exportChartPng(hotspotsRef, `hotspots_${hotForce}.png`)}
          >
            Export PNG
          </button>
        </div>
        {hotError && <div className="error-message">{hotError}</div>}
        {!hotLoading && hotspotsLoaded && hotspots.length === 0 && !hotError && (
          <div className="no-data">No hotspots found for that force.</div>
        )}
        {hotspots.length > 0 && (
          <div className="charts-grid">
            <div className="chart-container" ref={hotspotsRef}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hotspots.slice(0, 20)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="LSOA name" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="crime_count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="map-container">
              <MapContainer 
                center={(() => {
                  if (!hotForce) {
                    // Global view: center on UK
                    return [54.0, -2.0]; // UK center
                  }
                  // Force view: center on first valid hotspot
                  const validHotspot = hotspots.find(h => 
                    typeof h.Latitude === 'number' && 
                    typeof h.Longitude === 'number' &&
                    !isNaN(h.Latitude) && 
                    !isNaN(h.Longitude)
                  );
                  return validHotspot 
                    ? [validHotspot.Latitude, validHotspot.Longitude] 
                    : [52.3555, 0.1278]; // Cambridgeshire center
                })()} 
                zoom={hotForce ? 10 : 6} // Zoom out for global view
                style={{ height: '100%', width: '100%' }}
                key={hotspots.length} // Re-render map when hotspots change
              >
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
                {hotspots.map((h, i) => {
                  // Validate coordinates
                  const lat = parseFloat(h.Latitude);
                  const lng = parseFloat(h.Longitude);
                  
                  if (!isNaN(lat) && !isNaN(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
                    return (
                      <Marker key={i} position={[lat, lng]}>
                        <Popup>
                          <div>
                            <strong>{h['LSOA name'] || 'Unknown LSOA'}</strong>
                            <div>Crime Count: {h['crime_count'] || 0}</div>
                            {hotForce && <div>Force: {hotForce}</div>}
                          </div>
                        </Popup>
                      </Marker>
                    );
                  }
                  return null;
                })}
              </MapContainer>
            </div>
          </div>
        )}
      </Section>

      <Section title={`LSOA Time Series by Crime Type${hotForce ? ` (${hotForce})` : ''}`}>
        {hotForce && lsoaOptions.length > 0 && lsoa !== ALL_LSOAS_VALUE && (
          <p className="hint-text">Showing {lsoaOptions.length} LSOAs for {hotForce}. Select one to view its crime time series, or choose "All LSOAs in {hotForce}" for force-level trends.</p>
        )}
        {hotForce && lsoa === ALL_LSOAS_VALUE && (
          <p className="hint-text">Showing aggregated crime trends across all LSOAs in {hotForce}. Select a specific LSOA for detailed analysis.</p>
        )}
        {!hotForce && lsoaOptions.length > 0 && (
          <p className="hint-text">Showing all LSOAs across all forces. Select a force in the hotspots section above to filter LSOAs.</p>
        )}
        {lsoaOptions.length === 0 && (
          <p className="warning-text">No LSOAs available. Please select a force in the hotspots section above.</p>
        )}
        <div className="form-row">
          <select
            className="form-input"
            value={lsoa}
            onChange={e => setLsoa(e.target.value)}
            key={`${hotForce}-${lsoaOptions.length}`}
          >
            <option value="">Select LSOA...</option>
            {hotForce && (
              <option value={ALL_LSOAS_VALUE}>
                All LSOAs in {hotForce} (force-level trends)
              </option>
            )}
            {lsoaOptions.map(o => <option key={o} value={o}>{o}</option>)}
          </select>
          <button className="action-button" onClick={handleLoadSeries} disabled={seriesLoading}>
            {seriesLoading ? (
              <div className="loading-indicator">
                <div className="spinner"></div>
                Loading…
              </div>
            ) : 'Load'}
          </button>
          <button
            className="export-button"
            title={series.length === 0 ? 'Load data first' : ''}
            disabled={series.length === 0}
            onClick={() => downloadBlob(`lsoa_series_${lsoa}.csv`, toCsv(series), 'text/csv')}
          >
            Export CSV
          </button>
          <button
            className="export-button"
            title={series.length === 0 ? 'Load data first' : ''}
            disabled={series.length === 0}
            onClick={() => exportChartPng(seriesRef, `lsoa_series_${lsoa}.png`)}
          >
            Export PNG
          </button>
        </div>
        {seriesError && <div className="error-message">{seriesError}</div>}
        {!seriesLoading && seriesLoaded && seriesChartData.length === 0 && !seriesError && (
          <div className="no-data">No time series data for that LSOA.</div>
        )}
        {seriesChartData.length > 0 && (
          <div className="chart-container" ref={seriesRef}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={seriesChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="Month" />
                <YAxis />
                <Tooltip />
                <Legend />
                {crimeTypes.map((k, idx) => (
                  <Line key={k} type="monotone" dataKey={k} stroke={["#8884d8","#82ca9d","#ffc658","#ff7300","#00C49F"][idx % 5]} dot={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      <div className="footer">
        Confidential Law Enforcement Tool
      </div>
    </div>
  );
}
