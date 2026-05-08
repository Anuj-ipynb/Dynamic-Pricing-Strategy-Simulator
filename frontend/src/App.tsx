import { useEffect, useState } from "react"

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts"

const API_BASE = "http://127.0.0.1:8000"

export default function App() {

  // =====================================================
  // UI STATE
  // =====================================================
  const [selectedModel, setSelectedModel] = useState("ppo")

  const [loading, setLoading] = useState(false)

  const [steps, setSteps] = useState(100)

  // =====================================================
  // BACKEND DATA
  // =====================================================
  const [metrics, setMetrics] = useState<any>({})

  const [comparisonData, setComparisonData] = useState<any[]>([])

  const [trainingHistory, setTrainingHistory] = useState<any>(null)

  const [telemetry, setTelemetry] = useState<any>(null)

  const [modelInfo, setModelInfo] = useState<any>(null)

  const [experimentLogs, setExperimentLogs] = useState<any[]>([])

  // =====================================================
  // SIMULATION DATA
  // =====================================================
  const [simulationRevenue, setSimulationRevenue] = useState(0)

  const [simulationReward, setSimulationReward] = useState(0)

  const [priceVolatility, setPriceVolatility] = useState(0)

  const [avgPrice, setAvgPrice] = useState(0)

  const [pricingTrace, setPricingTrace] = useState<any[]>([])

  const [revenueTrace, setRevenueTrace] = useState<any[]>([])

  const [inventoryTrace, setInventoryTrace] = useState<any[]>([])

  const [trustTrace, setTrustTrace] = useState<any[]>([])

  // =====================================================
  // INITIAL LOAD
  // =====================================================
  useEffect(() => {

    loadDashboard()

  }, [])

  // =====================================================
  // LOAD DASHBOARD
  // =====================================================
  const loadDashboard = async () => {

    try {

      const [
        metricsRes,
        compareRes,
        historyRes,
        telemetryRes,
        metadataRes,
        experimentRes
      ] = await Promise.all([

        fetch(`${API_BASE}/metrics`),

        fetch(`${API_BASE}/compare`),

        fetch(`${API_BASE}/training-history`),

        fetch(`${API_BASE}/telemetry`),

        fetch(`${API_BASE}/model-info`),

        fetch(`${API_BASE}/experiment-log`)
      ])

      // -------------------------------------------------
      // JSON
      // -------------------------------------------------
      const metricsData = await metricsRes.json()

      const compareData = await compareRes.json()

      const historyData = await historyRes.json()

      const telemetryData = await telemetryRes.json()

      const metadataData = await metadataRes.json()

      const experimentData = await experimentRes.json()

      // -------------------------------------------------
      // STATE
      // -------------------------------------------------
      setMetrics(metricsData)

      setComparisonData(compareData.rankings)

      setTrainingHistory(historyData)

      setTelemetry(telemetryData)

      setModelInfo(metadataData)

      setExperimentLogs(experimentData)

    } catch (error) {

      console.error(error)

      alert("Failed to load dashboard data")
    }
  }

  // =====================================================
  // SIMULATION
  // =====================================================
  const runSimulation = async () => {

    try {

      setLoading(true)

      const response = await fetch(
        `${API_BASE}/simulate`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            steps,
            model: selectedModel,
          }),
        }
      )

      const data = await response.json()

      // -------------------------------------------------
      // KPI
      // -------------------------------------------------
      setSimulationRevenue(data.total_revenue)

      setSimulationReward(data.total_reward)

      setPriceVolatility(data.price_volatility)

      setAvgPrice(data.avg_price)

      // -------------------------------------------------
      // CHARTS
      // -------------------------------------------------
      setPricingTrace(
        data.pricing_trace.map(
          (p: number, idx: number) => ({
            step: idx + 1,
            price: p,
          })
        )
      )

      setRevenueTrace(
        data.revenue_trace.map(
          (r: number, idx: number) => ({
            step: idx + 1,
            revenue: r,
          })
        )
      )

      setInventoryTrace(
        data.inventory_trace.map(
          (i: number, idx: number) => ({
            step: idx + 1,
            inventory: i,
          })
        )
      )

      setTrustTrace(
        data.trust_trace.map(
          (t: number, idx: number) => ({
            step: idx + 1,
            trust: t,
          })
        )
      )

      setLoading(false)

    } catch (error) {

      console.error(error)

      alert("Simulation failed")

      setLoading(false)
    }
  }

  // =====================================================
  // TRAINING CURVE DATA
  // =====================================================
  const trainingCurveData =
    trainingHistory
      ? trainingHistory.ppo_rewards.map(
          (r: number, idx: number) => ({
            episode: idx + 1,
            ppo: r,
            dqn: trainingHistory.dqn_rewards[idx] || 0,
          })
        )
      : []

  // =====================================================
  // PIE DATA
  // =====================================================
  const telemetryData = telemetry
    ? [
        {
          name: "PPO",
          value: telemetry.models.ppo.loaded ? 1 : 0,
        },
        {
          name: "DQN",
          value: telemetry.models.dqn.loaded ? 1 : 0,
        },
      ]
    : []

  // =====================================================
  // UI
  // =====================================================
  return (

    <div className="min-h-screen bg-[#f5f7fb] text-[#111827]">

      <div className="max-w-7xl mx-auto px-8 py-8">

        {/* ================================================= */}
        {/* HEADER */}
        {/* ================================================= */}
        <div className="flex items-center justify-between mb-8">

          <div>

            <h1 className="text-4xl font-semibold tracking-tight">
              Dynamic Pricing Strategy Simulator
            </h1>

            <p className="text-gray-500 mt-2 text-lg">
              Reinforcement Learning Analytics & Monitoring Platform
            </p>

          </div>

          <div className="flex gap-3">

            <button
              onClick={runSimulation}
              className="bg-[#111827] text-white px-5 py-3 rounded-2xl hover:bg-[#1f2937] transition"
            >
              {loading ? "Running..." : "Execute Simulation"}
            </button>

            <button
              onClick={loadDashboard}
              className="border border-gray-300 px-5 py-3 rounded-2xl bg-white hover:bg-gray-100 transition"
            >
              Refresh Dashboard
            </button>

          </div>

        </div>

        {/* ================================================= */}
        {/* GRID */}
        {/* ================================================= */}
        <div className="grid grid-cols-12 gap-6">

          {/* ================================================= */}
          {/* SIDEBAR */}
          {/* ================================================= */}
          <div className="col-span-3 space-y-6">

            {/* ------------------------------------------------ */}
            {/* CONTROLS */}
            {/* ------------------------------------------------ */}
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
                    onChange={(e) =>
                      setSelectedModel(e.target.value)
                    }
                    className="w-full mt-2 rounded-2xl border border-gray-300 p-3 bg-white outline-none"
                  >
                    <option value="ppo">
                      PPO
                    </option>

                    <option value="dqn">
                      DQN
                    </option>
                  </select>

                </div>

                <div>

                  <label className="text-sm text-gray-500">
                    Simulation Horizon
                  </label>

                  <input
                    type="range"
                    min="20"
                    max="200"
                    value={steps}
                    onChange={(e) =>
                      setSteps(
                        Number(e.target.value)
                      )
                    }
                    className="w-full mt-3"
                  />

                  <div className="text-right text-sm text-gray-600 mt-1">
                    {steps} Steps
                  </div>

                </div>

              </div>

            </div>

            {/* ------------------------------------------------ */}
            {/* TELEMETRY */}
            {/* ------------------------------------------------ */}
            <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

              <h2 className="text-xl font-semibold mb-5">
                System Telemetry
              </h2>

              <div className="space-y-4 text-sm">

                <MetricRow
                  label="Platform"
                  value="Operational"
                />

                <MetricRow
                  label="PPO Status"
                  value={
                    telemetry?.models?.ppo?.loaded
                      ? "Loaded"
                      : "Unavailable"
                  }
                />

                <MetricRow
                  label="DQN Status"
                  value={
                    telemetry?.models?.dqn?.loaded
                      ? "Loaded"
                      : "Unavailable"
                  }
                />

                <MetricRow
                  label="Artifacts"
                  value="Tracked"
                />

              </div>

            </div>

            {/* ------------------------------------------------ */}
            {/* MODEL DISTRIBUTION */}
            {/* ------------------------------------------------ */}
            <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

              <h2 className="text-xl font-semibold mb-5">
                Deployment Health
              </h2>

              <div className="h-[240px]">

                <ResponsiveContainer width="100%" height="100%">

                  <PieChart>

                    <Pie
                      data={telemetryData}
                      dataKey="value"
                      nameKey="name"
                      outerRadius={80}
                    >

                      <Cell fill="#111827" />

                      <Cell fill="#6b7280" />

                    </Pie>

                    <Tooltip />

                    <Legend />

                  </PieChart>

                </ResponsiveContainer>

              </div>

            </div>

          </div>

          {/* ================================================= */}
          {/* MAIN */}
          {/* ================================================= */}
          <div className="col-span-9 space-y-6">

            {/* ================================================= */}
            {/* KPI */}
            {/* ================================================= */}
            <div className="grid grid-cols-4 gap-5">

              <KPI
                title="Simulation Revenue"
                value={`$${simulationRevenue.toFixed(0)}`}
              />

              <KPI
                title="Total Reward"
                value={simulationReward.toFixed(1)}
              />

              <KPI
                title="Avg Price"
                value={`$${avgPrice.toFixed(2)}`}
              />

              <KPI
                title="Price Volatility"
                value={priceVolatility.toFixed(2)}
              />

            </div>

            {/* ================================================= */}
            {/* TRAINING CURVES */}
            {/* ================================================= */}
            <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

              <h3 className="text-xl font-semibold mb-5">
                PPO vs DQN Reward Convergence
              </h3>

              <div className="h-[350px]">

                <ResponsiveContainer width="100%" height="100%">

                  <LineChart data={trainingCurveData}>

                    <CartesianGrid strokeDasharray="3 3" />

                    <XAxis dataKey="episode" />

                    <YAxis />

                    <Tooltip />

                    <Line
                      type="monotone"
                      dataKey="ppo"
                      stroke="#111827"
                      strokeWidth={3}
                      dot={false}
                    />

                    <Line
                      type="monotone"
                      dataKey="dqn"
                      stroke="#6b7280"
                      strokeWidth={3}
                      dot={false}
                    />

                  </LineChart>

                </ResponsiveContainer>

              </div>

            </div>

            {/* ================================================= */}
            {/* MAIN CHART GRID */}
            {/* ================================================= */}
            <div className="grid grid-cols-2 gap-6">

              {/* ------------------------------------------------ */}
              {/* PRICING */}
              {/* ------------------------------------------------ */}
              <ChartCard
                title="Dynamic Pricing Trajectory"
              >

                <ResponsiveContainer width="100%" height="100%">

                  <AreaChart data={pricingTrace}>

                    <CartesianGrid strokeDasharray="3 3" />

                    <XAxis dataKey="step" />

                    <YAxis />

                    <Tooltip />

                    <Area
                      type="monotone"
                      dataKey="price"
                      stroke="#111827"
                      fill="#d1d5db"
                      strokeWidth={2}
                    />

                  </AreaChart>

                </ResponsiveContainer>

              </ChartCard>

              {/* ------------------------------------------------ */}
              {/* INVENTORY */}
              {/* ------------------------------------------------ */}
              <ChartCard
                title="Inventory Depletion Curve"
              >

                <ResponsiveContainer width="100%" height="100%">

                  <LineChart data={inventoryTrace}>

                    <CartesianGrid strokeDasharray="3 3" />

                    <XAxis dataKey="step" />

                    <YAxis />

                    <Tooltip />

                    <Line
                      type="monotone"
                      dataKey="inventory"
                      stroke="#111827"
                      strokeWidth={3}
                      dot={false}
                    />

                  </LineChart>

                </ResponsiveContainer>

              </ChartCard>

              {/* ------------------------------------------------ */}
              {/* REVENUE */}
              {/* ------------------------------------------------ */}
              <ChartCard
                title="Revenue Trajectory"
              >

                <ResponsiveContainer width="100%" height="100%">

                  <AreaChart data={revenueTrace}>

                    <CartesianGrid strokeDasharray="3 3" />

                    <XAxis dataKey="step" />

                    <YAxis />

                    <Tooltip />

                    <Area
                      type="monotone"
                      dataKey="revenue"
                      stroke="#374151"
                      fill="#d1d5db"
                    />

                  </AreaChart>

                </ResponsiveContainer>

              </ChartCard>

              {/* ------------------------------------------------ */}
              {/* TRUST */}
              {/* ------------------------------------------------ */}
              <ChartCard
                title="Customer Trust Dynamics"
              >

                <ResponsiveContainer width="100%" height="100%">

                  <LineChart data={trustTrace}>

                    <CartesianGrid strokeDasharray="3 3" />

                    <XAxis dataKey="step" />

                    <YAxis />

                    <Tooltip />

                    <Line
                      type="monotone"
                      dataKey="trust"
                      stroke="#111827"
                      strokeWidth={3}
                      dot={false}
                    />

                  </LineChart>

                </ResponsiveContainer>

              </ChartCard>

            </div>

            {/* ================================================= */}
            {/* BOTTOM GRID */}
            {/* ================================================= */}
            <div className="grid grid-cols-3 gap-6">

              {/* ------------------------------------------------ */}
              {/* RL VS BASELINES */}
              {/* ------------------------------------------------ */}
              <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

                <h3 className="text-xl font-semibold mb-5">
                  RL vs Baselines
                </h3>

                <div className="h-[260px]">

                  <ResponsiveContainer width="100%" height="100%">

                    <BarChart data={comparisonData}>

                      <CartesianGrid strokeDasharray="3 3" />

                      <XAxis dataKey="model" />

                      <YAxis />

                      <Tooltip />

                      <Bar
                        dataKey="revenue"
                        fill="#111827"
                        radius={[8, 8, 0, 0]}
                      />

                    </BarChart>

                  </ResponsiveContainer>

                </div>

              </div>

              {/* ------------------------------------------------ */}
              {/* MODEL INFO */}
              {/* ------------------------------------------------ */}
              <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

                <h3 className="text-xl font-semibold mb-5">
                  Experiment Metadata
                </h3>

                <div className="space-y-4 text-sm">

                  <MetricRow
                    label="Run ID"
                    value={
                      modelInfo?.run_id
                        ?.slice(0, 10) || "N/A"
                    }
                  />

                  <MetricRow
                    label="Episodes"
                    value={
                      modelInfo?.episodes || "N/A"
                    }
                  />

                  <MetricRow
                    label="Seed"
                    value={
                      modelInfo?.seed || "N/A"
                    }
                  />

                  <MetricRow
                    label="PPO LR"
                    value={
                      modelInfo?.ppo_config
                        ?.learning_rate || "N/A"
                    }
                  />

                  <MetricRow
                    label="DQN LR"
                    value={
                      modelInfo?.dqn_config
                        ?.learning_rate || "N/A"
                    }
                  />

                </div>

              </div>

              {/* ------------------------------------------------ */}
              {/* LOGS */}
              {/* ------------------------------------------------ */}
              <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

                <h3 className="text-xl font-semibold mb-5">
                  RL Runtime Logs
                </h3>

                <div className="bg-[#f9fafb] border border-gray-200 rounded-2xl p-4 text-sm font-mono h-[260px] overflow-auto text-gray-700 space-y-2">

                  {experimentLogs.map((log, idx) => (

                    <div key={idx}>

                      [{log.model}] Ep {log.episode}
                      {" | "}
                      Reward:
                      {" "}
                      {Number(log.reward).toFixed(2)}

                    </div>

                  ))}

                </div>

              </div>

            </div>

          </div>

        </div>

      </div>

    </div>
  )
}

// =========================================================
// KPI
// =========================================================
function KPI({ title, value }: any) {

  return (

    <div className="bg-white rounded-3xl border border-gray-200 p-5 shadow-sm">

      <div className="text-sm text-gray-500">
        {title}
      </div>

      <div className="text-3xl font-semibold mt-3">
        {value}
      </div>

    </div>
  )
}

// =========================================================
// METRIC ROW
// =========================================================
function MetricRow({ label, value }: any) {

  return (

    <div className="flex justify-between border-b border-gray-100 pb-3">

      <span className="text-gray-500">
        {label}
      </span>

      <span className="font-medium">
        {value}
      </span>

    </div>
  )
}

// =========================================================
// CHART CARD
// =========================================================
function ChartCard({ title, children }: any) {

  return (

    <div className="bg-white rounded-3xl border border-gray-200 p-6 shadow-sm">

      <h3 className="text-xl font-semibold mb-5">
        {title}
      </h3>

      <div className="h-[280px]">
        {children}
      </div>

    </div>
  )
}