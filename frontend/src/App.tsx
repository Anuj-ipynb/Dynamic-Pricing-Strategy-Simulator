import { useState } from "react"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from "recharts"

export default function App() {
  const [selectedModel, setSelectedModel] = useState("ppo")
  const [loading, setLoading] = useState(false)

  const [pricingData, setPricingData] = useState<any[]>([])
  const [revenue, setRevenue] = useState(0)

  const comparisonData = [
    { model: "PPO", revenue: 18100 },
    { model: "DQN", revenue: 15200 },
    { model: "Rule", revenue: 11800 },
    { model: "Bandit", revenue: 9700 },
  ]

  const revenueData = [
    { episode: 100, revenue: 4200 },
    { episode: 200, revenue: 6700 },
    { episode: 300, revenue: 9200 },
    { episode: 400, revenue: 11800 },
    { episode: 500, revenue: 13200 },
    { episode: 600, revenue: 14500 },
    { episode: 700, revenue: 15800 },
    { episode: 800, revenue: 16700 },
    { episode: 900, revenue: 17300 },
    { episode: 1000, revenue: 18100 },
  ]

  const runSimulation = async () => {
    try {
      setLoading(true)

      const response = await fetch("http://127.0.0.1:8000/simulate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          steps: 100,
          model: selectedModel,
        }),
      })

      const data = await response.json()

      const transformed = data.prices.map((p: number, idx: number) => ({
        step: idx + 1,
        price: p,
      }))

      setPricingData(transformed)
      setRevenue(data.total_revenue)

      setLoading(false)
    } catch (error) {
      console.error(error)
      alert("Simulation failed")
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#f5f7fb] text-[#111827]">
      <div className="max-w-7xl mx-auto px-8 py-8">

        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight">
              Dynamic Pricing Strategy Simulator
            </h1>

            <p className="text-gray-500 mt-2 text-lg">
              Reinforcement Learning Marketplace Intelligence Platform
            </p>
          </div>

          <div className="flex gap-3">
            <button
              onClick={runSimulation}
              className="bg-[#111827] text-white px-5 py-3 rounded-2xl hover:bg-[#1f2937] transition"
            >
              {loading ? "Running..." : "Execute Simulation"}
            </button>

            <button className="border border-gray-300 px-5 py-3 rounded-2xl bg-white hover:bg-gray-100 transition">
              Export Results
            </button>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-6">

          {/* Sidebar */}
          <div className="col-span-3 space-y-6">

            <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">
              <h2 className="text-xl font-semibold mb-6">
                Simulation Controls
              </h2>

              <div className="space-y-5">

                <div>
                  <label className="text-sm text-gray-500">
                    RL Agent
                  </label>

                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full mt-2 rounded-2xl border border-gray-300 p-3 bg-white outline-none"
                  >
                    <option value="ppo">PPO</option>
                    <option value="dqn">DQN</option>
                  </select>
                </div>

                <div>
                  <label className="text-sm text-gray-500">
                    Simulation Horizon
                  </label>

                  <input
                    type="range"
                    min="10"
                    max="200"
                    defaultValue="100"
                    className="w-full mt-3"
                  />

                  <div className="text-right text-sm text-gray-600 mt-1">
                    100 Steps
                  </div>
                </div>

              </div>
            </div>

            <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">
              <h2 className="text-xl font-semibold mb-5">
                Environment Metrics
              </h2>

              <div className="space-y-4 text-sm">
                <MetricRow label="Current Model" value={selectedModel.toUpperCase()} />
                <MetricRow label="Demand Noise" value="4.0" />
                <MetricRow label="Price Elasticity" value="2.2" />
                <MetricRow label="Reward Stability" value="Stable" />
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="col-span-9 space-y-6">

            {/* KPI Cards */}
            <div className="grid grid-cols-4 gap-5">
              <KPI title="Simulation Revenue" value={`$${revenue.toFixed(0)}`} />
              <KPI title="RL Agent" value={selectedModel.toUpperCase()} />
              <KPI title="Episodes" value="1000" />
              <KPI title="Environment" value="Non-Stationary" />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-2 gap-6">

              {/* Revenue Chart */}
              <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

                <h3 className="text-xl font-semibold mb-4">
                  PPO Training Curve
                </h3>

                <div className="h-[280px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={revenueData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="episode" />
                      <YAxis />
                      <Tooltip />

                      <Line
                        type="monotone"
                        dataKey="revenue"
                        stroke="#111827"
                        strokeWidth={3}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Dynamic Pricing Chart */}
              <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

                <h3 className="text-xl font-semibold mb-4">
                  Dynamic Pricing Trajectory
                </h3>

                <div className="h-[280px]">
                  <ResponsiveContainer width="100%" height="100%">

                    <AreaChart data={pricingData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="step" />
                      <YAxis />
                      <Tooltip />

                      <Area
                        type="monotone"
                        dataKey="price"
                        stroke="#374151"
                        fill="#d1d5db"
                        strokeWidth={2}
                      />
                    </AreaChart>

                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Bottom Grid */}
            <div className="grid grid-cols-3 gap-6">

              <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

                <h3 className="text-xl font-semibold mb-5">
                  RL vs Baselines
                </h3>

                <div className="h-[240px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="model" />
                      <YAxis />
                      <Tooltip />

                      <Bar
                        dataKey="revenue"
                        fill="#374151"
                        radius={[8, 8, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

                <h3 className="text-xl font-semibold mb-5">
                  MLOps Pipeline
                </h3>

                <div className="space-y-4 text-sm">
                  <PipelineStatus label="Training Pipeline" />
                  <PipelineStatus label="Experiment Tracking" />
                  <PipelineStatus label="Checkpointing" />
                  <PipelineStatus label="FastAPI Service" />
                  <PipelineStatus label="Evaluation Suite" />
                </div>
              </div>

              <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

                <h3 className="text-xl font-semibold mb-5">
                  RL Runtime Logs
                </h3>

                <div className="bg-[#f9fafb] border border-gray-200 rounded-2xl p-4 text-sm font-mono h-[240px] overflow-auto text-gray-700 space-y-2">
                  <div>Simulation executed successfully</div>
                  <div>Model: {selectedModel.toUpperCase()}</div>
                  <div>Total Revenue: ${revenue.toFixed(2)}</div>
                  <div>Environment: Non-Stationary</div>
                  <div>Adaptive Pricing Active</div>
                </div>

              </div>

            </div>

          </div>
        </div>
      </div>
    </div>
  )
}

function KPI({ title, value }: any) {
  return (
    <div className="bg-white rounded-3xl border border-gray-200 p-5 shadow-sm">
      <div className="text-sm text-gray-500">{title}</div>
      <div className="text-3xl font-semibold mt-3">{value}</div>
    </div>
  )
}

function MetricRow({ label, value }: any) {
  return (
    <div className="flex justify-between border-b border-gray-100 pb-3">
      <span className="text-gray-500">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  )
}

function PipelineStatus({ label }: any) {
  return (
    <div className="flex items-center justify-between border-b border-gray-100 pb-3">
      <span className="text-gray-500">{label}</span>

      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-green-500" />
        <span className="text-sm">Operational</span>
      </div>
    </div>
  )
}