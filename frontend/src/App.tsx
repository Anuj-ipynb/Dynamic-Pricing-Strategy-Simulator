import React, { useState } from 'react';

// Interfaces for structured API typings
interface SimulationResponse {
  prices: number[];
  total_revenue: number;
  steps_executed: number;
}

interface MetricSummary {
  "Total Revenue": number;
  "Total Sales Units": number;
  "Mean Applied Price": number;
  "Price Volatility (StdDev)": number;
  "Final Terminal Inventory": number;
}

interface GlobalMetrics {
  [modelName: string]: MetricSummary;
}

export default function App() {
  const [model, setModel] = useState<string>('ppo');
  const [steps, setSteps] = useState<number>(100);
  const [simData, setSimData] = useState<SimulationResponse | null>(null);
  const [metrics, setMetrics] = useState<GlobalMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const BACKEND_URL = 'http://localhost:8000';

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps, model }),
      });
      if (!response.ok) throw new Error('Simulation failed on the backend cluster.');
      const data: SimulationResponse = await response.json();
      setSimData(data);
    } catch (err: any) {
      setError(err.message || 'Network connectivity error.');
    } finally {
      setLoading(false);
    }
  };

  const fetchGlobalMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/metrics`);
      if (!response.ok) throw new Error('Metrics logs not generated yet. Run evaluate.py first.');
      const data: GlobalMetrics = await response.json();
      setMetrics(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch global metrics.');
    } finally {
      setLoading(false);
    }
  };

  // Inline Pure SVG Chart Generator Component
  const SvgLineChart = ({ data }: { data: number[] }) => {
    if (!data || data.length === 0) return null;
    const width = 600;
    const height = 200;
    const padding = 20;
    
    const maxVal = Math.max(...data, 100);
    const minVal = Math.min(...data, 0);
    const valRange = maxVal - minVal || 1;

    const points = data.map((val, index) => {
      const x = padding + (index / (data.length - 1)) * (width - padding * 2);
      const y = height - padding - ((val - minVal) / valRange) * (height - padding * 2);
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ background: '#1e293b', borderRadius: '8px' }}>
        <polyline fill="none" stroke="#10b981" strokeWidth="3" points={points} />
      </svg>
    );
  };

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ borderBottom: '1px solid #334155', paddingBottom: '1rem', marginBottom: '2rem' }}>
        <h1 style={{ margin: 0, color: '#38bdf8' }}>Dynamic Pricing Strategy Control Simulator</h1>
        <p style={{ color: '#94a3b8', margin: '0.5rem 0 0 0' }}>Reinforcement Learning Engine Production Shell</p>
      </header>

      {error && (
        <div style={{ background: '#ef444422', border: '1px solid #ef4444', color: '#fca5a5', padding: '1rem', borderRadius: '6px', marginBottom: '1.5rem' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: '2rem' }}>
        {/* Controls Panel */}
        <aside style={{ background: '#1e293b', padding: '1.5rem', borderRadius: '8px', height: 'fit-content' }}>
          <h3 style={{ marginTop: 0, borderBottom: '1px solid #475569', paddingBottom: '0.5rem' }}>Configuration</h3>
          
          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#94a3b8' }}>Target Execution Agent</label>
            <select 
              value={model} 
              onChange={(e) => setModel(e.target.value)}
              style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', background: '#0f172a', color: '#fff', border: '1px solid #475569' }}
            >
              <option value="ppo">Continuous PPO</option>
              <option value="dqn">Discrete DQN</option>
            </select>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#94a3b8' }}>Simulation Horizon: {steps}</label>
            <input 
              type="range" min="10" max="100" value={steps} 
              onChange={(e) => setSteps(Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <button 
            onClick={runSimulation} 
            disabled={loading}
            style={{ width: '100%', padding: '0.75rem', borderRadius: '4px', background: '#2563eb', color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 'bold', marginBottom: '1rem' }}
          >
            {loading ? 'Processing...' : 'Execute Simulation'}
          </button>

          <button 
            onClick={fetchGlobalMetrics} 
            disabled={loading}
            style={{ width: '100%', padding: '0.75rem', borderRadius: '4px', background: '#10b981', color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}
          >
            Fetch Performance Ledger
          </button>
        </aside>

        {/* Live Metrics Display Areas */}
        <main style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {simData && (
            <div style={{ background: '#1e293b', padding: '1.5rem', borderRadius: '8px' }}>
              <h2 style={{ marginTop: 0, color: '#10b981' }}>Live Simulation Output Traces</h2>
              <div style={{ display: 'flex', gap: '2rem', marginBottom: '1.5rem' }}>
                <div>
                  <small style={{ color: '#94a3b8', display: 'block' }}>Total Generated Revenue</small>
                  <span style={{ fontSize: '1.75rem', fontWeight: 'bold', color: '#38bdf8' }}>${simData.total_revenue.toFixed(2)}</span>
                </div>
                <div>
                  <small style={{ color: '#94a3b8', display: 'block' }}>Steps Logged</small>
                  <span style={{ fontSize: '1.75rem', fontWeight: 'bold' }}>{simData.steps_executed}</span>
                </div>
              </div>
              <h3>Pricing Path Array Traced</h3>
              <SvgLineChart data={simData.prices} />
            </div>
          )}

          {metrics && (
            <div style={{ background: '#1e293b', padding: '1.5rem', borderRadius: '8px', overflowX: 'auto' }}>
              <h2 style={{ marginTop: 0, color: '#38bdf8' }}>Comparative Cross-Model Leaderboard</h2>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #475569', color: '#94a3b8' }}>
                    <th style={{ padding: '0.75rem' }}>Architecture Variant</th>
                    <th style={{ padding: '0.75rem' }}>Total Revenue</th>
                    <th style={{ padding: '0.75rem' }}>Units Sold</th>
                    <th style={{ padding: '0.75rem' }}>Mean Price</th>
                    <th style={{ padding: '0.75rem' }}>Price Volatility (StdDev)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(metrics).map((modelName) => (
                    <tr key={modelName} style={{ borderBottom: '1px solid #334155' }}>
                      <td style={{ padding: '0.75rem', fontWeight: 'bold' }}>{modelName}</td>
                      <td style={{ padding: '0.75rem', color: '#10b981', fontWeight: 'bold' }}>${metrics[modelName]["Total Revenue"].toFixed(2)}</td>
                      <td style={{ padding: '0.75rem' }}>{metrics[modelName]["Total Sales Units"].toFixed(0)}</td>
                      <td style={{ padding: '0.75rem' }}>${metrics[modelName]["Mean Applied Price"].toFixed(2)}</td>
                      <td style={{ padding: '0.75rem', color: metrics[modelName]["Price Volatility (StdDev)"] > 15 ? '#f87171' : '#38bdf8' }}>
                        {metrics[modelName]["Price Volatility (StdDev)"].toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}