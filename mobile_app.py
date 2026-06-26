from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request

from src.config import API_URL, CSV_SEPARATOR, DATA_PATH, DATE_COLUMN, MODEL_DIR, PRICE_COLUMN
from src.dynamic_pricing import load_pricing_data, predict_future_prices


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "data" / "electronics_price_history_sample.json"

app = Flask(__name__)


MOBILE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pokemon Dynamic Pricing</title>
  <style>
    :root {
      --bg: #f4f6fb;
      --panel: #ffffff;
      --text: #111827;
      --muted: #64748b;
      --line: #e5e7eb;
      --primary: #4f46e5;
      --primary-dark: #3730a3;
      --success: #047857;
      --danger: #dc2626;
      --warning: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell {
      width: min(100%, 460px);
      min-height: 100vh;
      margin: 0 auto;
      padding: 18px 14px 92px;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
    }
    .brand h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .brand p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .pill {
      border-radius: 999px;
      background: #eef2ff;
      color: var(--primary-dark);
      padding: 8px 10px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
      margin-bottom: 12px;
    }
    .hero {
      background: linear-gradient(135deg, #4338ca, #7c3aed 55%, #db2777);
      color: #ffffff;
      border: 0;
    }
    .hero .muted { color: rgba(255,255,255,0.78); }
    .hero-price {
      margin: 10px 0 0;
      font-size: 42px;
      font-weight: 800;
      letter-spacing: 0;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 13px;
      min-height: 104px;
    }
    .metric label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .metric strong {
      display: block;
      font-size: 23px;
      line-height: 1.1;
      word-break: break-word;
    }
    .metric span {
      display: block;
      margin-top: 7px;
      font-size: 12px;
      color: var(--muted);
    }
    h2 {
      margin: 0 0 12px;
      font-size: 17px;
      letter-spacing: 0;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 700;
      margin: 12px 0 6px;
    }
    input, select, textarea, button {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      font: inherit;
    }
    input, select, textarea {
      background: #ffffff;
      color: var(--text);
      padding: 12px;
      outline: none;
    }
    textarea {
      min-height: 150px;
      resize: vertical;
    }
    button {
      border: 0;
      background: var(--primary);
      color: #ffffff;
      font-weight: 800;
      padding: 13px 14px;
      margin-top: 12px;
      cursor: pointer;
    }
    button.secondary {
      background: #eef2ff;
      color: var(--primary-dark);
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .list-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding: 10px 0;
      font-size: 14px;
      gap: 10px;
    }
    .list-row:last-child { border-bottom: 0; }
    .muted {
      color: var(--muted);
      font-size: 13px;
    }
    .up { color: var(--success) !important; }
    .down { color: var(--danger) !important; }
    .warn { color: var(--warning) !important; }
    .error {
      color: #991b1b;
      background: #fee2e2;
      border: 1px solid #fecaca;
      border-radius: 12px;
      padding: 10px;
      margin-top: 10px;
      font-size: 13px;
    }
    .success-box {
      color: #065f46;
      background: #d1fae5;
      border: 1px solid #a7f3d0;
      border-radius: 12px;
      padding: 10px;
      margin-top: 10px;
      font-size: 13px;
    }
    .tabs {
      position: fixed;
      left: 50%;
      bottom: 0;
      transform: translateX(-50%);
      width: min(100%, 460px);
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      padding: 10px 12px 14px;
      background: rgba(255, 255, 255, 0.94);
      border-top: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }
    .tab {
      margin: 0;
      border-radius: 999px;
      background: #f1f5f9;
      color: var(--muted);
      padding: 10px 8px;
      font-size: 13px;
    }
    .tab.active {
      background: var(--primary);
      color: #ffffff;
    }
    .screen { display: none; }
    .screen.active { display: block; }
    .login-screen {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: grid;
      place-items: center;
      background: linear-gradient(160deg, #eef2ff, #f8fafc 52%, #fce7f3);
      padding: 18px;
    }
    .login-screen.hidden { display: none; }
    .login-card {
      width: min(100%, 420px);
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.16);
    }
    .login-title {
      margin: 0;
      font-size: 30px;
      letter-spacing: 0;
    }
    .role-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 16px;
    }
    .role-card {
      min-height: 132px;
      margin: 0;
      text-align: left;
      border: 1px solid var(--line);
      background: #f8fafc;
      color: var(--text);
      padding: 14px;
    }
    .role-card strong {
      display: block;
      font-size: 18px;
      margin-bottom: 8px;
    }
    .role-card span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    .role-card.primary {
      background: var(--primary);
      color: #ffffff;
    }
    .role-card.primary span {
      color: rgba(255,255,255,0.78);
    }
    .logout-button {
      width: auto;
      margin: 0;
      padding: 8px 10px;
      background: #f1f5f9;
      color: var(--muted);
      font-size: 12px;
    }
    .chart {
      width: 100%;
      height: 220px;
      display: block;
    }
  </style>
</head>
<body data-default-api="{{ default_api_url }}">
  <section id="loginScreen" class="login-screen">
    <div class="login-card">
      <p class="muted">Collektr AI pricing</p>
      <h1 class="login-title">Choose your role</h1>
      <p class="muted">The mobile interface will open the tools that match the user type.</p>
      <div class="role-grid">
        <button class="role-card primary" data-role="buyer">
          <strong>Buyer</strong>
          <span>View market price, market low, and Pokemon price forecast.</span>
        </button>
        <button class="role-card" data-role="seller">
          <strong>Seller</strong>
          <span>Set minimum price, follow market price, and check seller alerts.</span>
        </button>
      </div>
    </div>
  </section>

  <main class="shell">
    <section class="topbar">
      <div class="brand">
        <h1>Pokemon Pricing</h1>
        <p id="screenSubtitle">Buyer market forecast</p>
      </div>
      <div>
        <div class="pill" id="statusPill">Ready</div>
        <button class="logout-button" id="logoutButton">Switch</button>
      </div>
    </section>

    <section id="screenBuyer" class="screen active">
      <div class="panel">
        <h2>Pokemon card</h2>
        <label>Card</label>
        <select id="pokemonCard"></select>
        <button id="pokemonPredictButton">Predict Pokemon price</button>
      </div>

      <div class="panel hero">
        <div class="muted">Buyer target price</div>
        <div class="hero-price" id="buyerTargetPrice">$0.00</div>
        <div class="muted" id="buyerNote">Select a card to view market movement.</div>
      </div>

      <div class="grid">
        <div class="metric">
          <label>Current market</label>
          <strong id="buyerLatestPrice">$0.00</strong>
          <span id="buyerLatestDate">No data</span>
        </div>
        <div class="metric">
          <label>Market low</label>
          <strong id="buyerMarketLow">$0.00</strong>
          <span>Lowest market signal</span>
        </div>
        <div class="metric">
          <label>Daily movement</label>
          <strong id="buyerDailyMove">$0.00</strong>
          <span id="buyerDailyPct">0.00%</span>
        </div>
        <div class="metric">
          <label>7-day forecast</label>
          <strong id="buyerForecastPrice">$0.00</strong>
          <span id="buyerForecastDelta">No prediction</span>
        </div>
      </div>

      <div class="panel" style="margin-top:12px;">
        <h2>Price graph</h2>
        <canvas id="buyerChart" class="chart" width="400" height="220"></canvas>
      </div>

      <div class="panel">
        <h2>Forecast</h2>
        <div id="buyerForecastList" class="muted">No forecast yet.</div>
      </div>
    </section>

    <section id="screenSeller" class="screen">
      <div class="panel">
        <h2>Pokemon card</h2>
        <label>Card</label>
        <select id="sellerPokemonCard"></select>
        <button id="sellerPredictButton">Predict seller price</button>
      </div>

      <div class="panel hero">
        <div class="muted">Suggested listing price</div>
        <div class="hero-price" id="sellerListingPrice">$0.00</div>
        <div class="muted" id="sellerNote">Seller price follows market, protected by minimum price.</div>
      </div>

      <div class="panel">
        <h2>Seller controls</h2>
        <div class="row">
          <div>
            <label>Minimum price</label>
            <input id="minimumPrice" type="number" min="0" step="0.01" value="30">
          </div>
          <div>
            <label>Cost price</label>
            <input id="costPrice" type="number" min="0" step="0.01" value="20">
          </div>
        </div>
        <button class="secondary" id="refreshSellerButton">Recalculate seller price</button>
      </div>

      <div class="grid">
        <div class="metric">
          <label>Current market</label>
          <strong id="sellerMarketPrice">$0.00</strong>
          <span>Latest Pokemon price</span>
        </div>
        <div class="metric">
          <label>Minimum price</label>
          <strong id="sellerMinimumPrice">$30.00</strong>
          <span>Seller floor</span>
        </div>
        <div class="metric">
          <label>Profit each</label>
          <strong id="sellerProfitEach">$0.00</strong>
          <span id="sellerProfitNote">Waiting</span>
        </div>
        <div class="metric">
          <label>Status</label>
          <strong id="sellerStatus">Waiting</strong>
          <span id="sellerStatusNote">Run prediction</span>
        </div>
      </div>

      <div class="panel" style="margin-top:12px;">
        <h2>Seller price graph</h2>
        <canvas id="sellerChart" class="chart" width="400" height="220"></canvas>
      </div>

      <div class="panel">
        <h2>Listing rule</h2>
        <div class="list-row">
          <span>Automatic listing price</span>
          <strong>max(market/forecast, minimum)</strong>
        </div>
        <div class="list-row">
          <span>Alert when forecast below minimum</span>
          <strong id="sellerAlert">No alert</strong>
        </div>
      </div>
    </section>

    <section id="screenCustom" class="screen">
      <div class="panel">
        <h2>Custom Data</h2>
        <label>API URL</label>
        <input id="apiUrl" type="url" placeholder="https://your-api-url">

        <button class="secondary" id="loadSampleButton">Use sample JSON data</button>

        <label>Upload JSON</label>
        <input id="jsonFile" type="file" accept="application/json,.json">

        <label>Or paste JSON history</label>
        <textarea id="jsonText" placeholder='[{"timestamp":"2026-01-01","item":"Product A","current_price":100}]'></textarea>
      </div>

      <div class="panel" id="mappingPanel" style="display:none;">
        <h2>Data mapping</h2>
        <label>Item column</label>
        <select id="itemColumn"></select>
        <label>Date column</label>
        <select id="dateColumn"></select>
        <label>Price column</label>
        <select id="priceColumn"></select>
        <label>Item</label>
        <select id="itemSelect"></select>
        <button id="customPredictButton">Train and predict custom data</button>
        <div id="dataMessage"></div>
      </div>

      <div class="panel">
        <h2>Model result</h2>
        <div class="grid">
          <div class="metric">
            <label>Best model</label>
            <strong id="bestModel">-</strong>
            <span>Prediction model</span>
          </div>
          <div class="metric">
            <label>Rows used</label>
            <strong id="historyRows">0</strong>
            <span>Training history</span>
          </div>
          <div class="metric">
            <label>Prediction time</label>
            <strong id="predictionTime">0.000s</strong>
            <span>API forecast</span>
          </div>
          <div class="metric">
            <label>Total time</label>
            <strong id="totalTime">0.000s</strong>
            <span>API request</span>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>API response</h2>
        <textarea id="apiResponse" readonly></textarea>
      </div>
    </section>
  </main>

  <nav class="tabs">
    <button class="tab active" data-screen="screenBuyer">Buyer</button>
    <button class="tab" data-screen="screenSeller">Seller</button>
    <button class="tab" data-screen="screenCustom">Custom Data</button>
  </nav>

  <script>
    const state = {
      pokemonCards: [],
      selectedCard: null,
      pokemonPrediction: null,
      records: [],
      customResult: null,
      customContext: null,
      role: null,
      selectedCardName: null
    };

    const subtitles = {
      screenBuyer: "Buyer market forecast",
      screenSeller: "Seller listing controls",
      screenCustom: "Upload custom history"
    };

    const $ = (id) => document.getElementById(id);
    const money = (value) => Number.isFinite(Number(value)) ? `$${Number(value).toFixed(2)}` : "-";
    const deltaMoney = (value) => {
      const number = Number(value);
      if (!Number.isFinite(number)) return "";
      return `${number >= 0 ? "+" : "-"}$${Math.abs(number).toFixed(2)}`;
    };

    function setStatus(text) {
      $("statusPill").textContent = text;
    }

    function showMessage(text, type = "success") {
      $("dataMessage").innerHTML = `<div class="${type === "error" ? "error" : "success-box"}">${text}</div>`;
    }

    function fillSelect(select, values, selectedValue) {
      select.innerHTML = "";
      values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        option.selected = value === selectedValue;
        select.appendChild(option);
      });
    }

    function selectedPokemonCard() {
      const name = state.selectedCardName || $("pokemonCard").value || $("sellerPokemonCard").value;
      return state.pokemonCards.find((card) => card.name === name) || null;
    }

    function syncPokemonSelectors(name) {
      state.selectedCardName = name;
      if ($("pokemonCard").value !== name) $("pokemonCard").value = name;
      if ($("sellerPokemonCard").value !== name) $("sellerPokemonCard").value = name;
    }

    function changePokemonCard(name) {
      syncPokemonSelectors(name);
      state.pokemonPrediction = null;
      renderPokemonMarket();
      $("buyerForecastList").textContent = "Run prediction to view the 7-day forecast.";
      renderCharts();
    }

    function setRole(role) {
      state.role = role;
      $("loginScreen").classList.add("hidden");
      document.querySelectorAll(".tab").forEach((tab) => {
        const screen = tab.dataset.screen;
        tab.style.display = "block";
        if (role === "buyer" && screen === "screenSeller") tab.style.display = "none";
        if (role === "seller" && screen === "screenBuyer") tab.style.display = "none";
      });
      openScreen(role === "seller" ? "screenSeller" : "screenBuyer");
      renderSeller();
      renderCharts();
    }

    function logout() {
      state.role = null;
      $("loginScreen").classList.remove("hidden");
    }

    function chartPoints(card, forecast) {
      const points = [];
      if (card) {
        if (Number.isFinite(Number(card.previous_market))) {
          points.push({ label: "Prev", value: Number(card.previous_market) });
        }
        points.push({ label: "Now", value: Number(card.market) });
      }
      forecast.forEach((row) => points.push({ label: `D${row.day}`, value: Number(row.predicted_price) }));
      return points.filter((point) => Number.isFinite(point.value));
    }

    function drawChart(canvasId, points, color, floorPrice = null) {
      const canvas = $(canvasId);
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      if (points.length < 2) {
        ctx.fillStyle = "#64748b";
        ctx.font = "14px system-ui";
        ctx.fillText("Run prediction to show graph", 18, height / 2);
        return;
      }

      const values = points.map((point) => point.value);
      if (Number.isFinite(Number(floorPrice))) values.push(Number(floorPrice));
      const min = Math.min(...values) * 0.96;
      const max = Math.max(...values) * 1.04;
      const range = max - min || 1;
      const padX = 36;
      const padY = 42;
      const plotW = width - padX * 2;
      const plotH = height - padY * 2;
      const xFor = (index) => padX + (plotW * index) / (points.length - 1);
      const yFor = (value) => padY + plotH - ((value - min) / range) * plotH;

      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i += 1) {
        const y = padY + (plotH * i) / 3;
        ctx.beginPath();
        ctx.moveTo(padX, y);
        ctx.lineTo(width - padX, y);
        ctx.stroke();
      }

      if (Number.isFinite(Number(floorPrice))) {
        const y = yFor(Number(floorPrice));
        ctx.strokeStyle = "#b45309";
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(padX, y);
        ctx.lineTo(width - padX, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#b45309";
        ctx.font = "bold 11px system-ui";
        ctx.textAlign = "left";
        ctx.fillText(`Min ${money(floorPrice)}`, padX + 4, Math.max(14, y - 6));
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.beginPath();
      points.forEach((point, index) => {
        const x = xFor(index);
        const y = yFor(point.value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      points.forEach((point, index) => {
        const x = xFor(index);
        const y = yFor(point.value);
        const priceText = money(point.value);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = "bold 11px system-ui";
        ctx.textAlign = "center";
        const labelWidth = ctx.measureText(priceText).width + 10;
        const labelX = Math.min(width - labelWidth / 2 - 4, Math.max(labelWidth / 2 + 4, x));
        const labelY = Math.max(14, y - 14);
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#cbd5e1";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(labelX - labelWidth / 2, labelY - 13, labelWidth, 18, 7);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#111827";
        ctx.fillText(priceText, labelX, labelY);

        ctx.fillStyle = "#64748b";
        ctx.font = "11px system-ui";
        ctx.textAlign = "center";
        ctx.fillText(point.label, x, height - 7);
      });
    }

    function renderCharts() {
      const card = selectedPokemonCard();
      const forecast = state.pokemonPrediction?.forecast || [];
      const points = chartPoints(card, forecast);
      drawChart("buyerChart", points, "#4f46e5");
      drawChart("sellerChart", points, "#047857", Number($("minimumPrice").value || 0));
    }

    async function loadPokemonCards() {
      try {
        const response = await fetch("/pokemon-cards");
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || "Could not load Pokemon card data.");
        state.pokemonCards = body.cards || [];
        const cardNames = state.pokemonCards.map((card) => card.name);
        fillSelect($("pokemonCard"), cardNames, body.default_card);
        fillSelect($("sellerPokemonCard"), cardNames, body.default_card);
        syncPokemonSelectors(body.default_card);
        renderPokemonMarket();
        setStatus("Pokemon loaded");
      } catch (error) {
        setStatus("Error");
        $("buyerNote").textContent = error.message;
      }
    }

    function renderPokemonMarket() {
      const card = selectedPokemonCard();
      if (!card) return;
      state.selectedCard = card;
      $("buyerLatestPrice").textContent = money(card.market);
      $("buyerLatestDate").textContent = card.updated_at || "No date";
      $("buyerMarketLow").textContent = money(card.market_low);
      $("buyerDailyMove").textContent = deltaMoney(card.change);
      $("buyerDailyMove").className = card.change >= 0 ? "up" : "down";
      $("buyerDailyPct").textContent = `${Number(card.change_pct || 0).toFixed(2)}%`;
      $("buyerDailyPct").className = card.change >= 0 ? "up" : "down";
      $("sellerMarketPrice").textContent = money(card.market);
      renderSeller();
      renderCharts();
    }

    async function predictPokemon() {
      const card = selectedPokemonCard();
      if (!card) return;
      const apiUrl = $("apiUrl").value.trim().replace(/\\/$/, "");
      $("pokemonPredictButton").disabled = true;
      setStatus("Predicting");
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_url: apiUrl, item: card.name, horizon: 7 })
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
        state.pokemonPrediction = body.predictions?.[0] || null;
        renderPokemonPrediction(body);
        setStatus("Complete");
      } catch (error) {
        setStatus("Error");
        $("buyerForecastList").innerHTML = `<div class="error">${error.message}</div>`;
      } finally {
        $("pokemonPredictButton").disabled = false;
      }
    }

    function renderPokemonPrediction(body) {
      const card = selectedPokemonCard();
      const prediction = state.pokemonPrediction;
      if (!card || !prediction) return;
      const forecast = prediction.forecast || [];
      const lastForecast = forecast.at(-1);
      const forecastPrice = Number(lastForecast?.predicted_price ?? prediction.predicted_price ?? 0);
      const delta = forecastPrice - Number(card.market);
      const targetBuyPrice = Math.min(Number(card.market), forecastPrice);
      const buyerSignal = forecastPrice < Number(card.market) ? "Buy later" : "Buy before rise";

      $("buyerTargetPrice").textContent = money(targetBuyPrice);
      $("buyerNote").textContent = `${card.name} forecast using ${String(prediction.model_name || "model").replace(/_/g, " ")}.`;
      $("buyerForecastPrice").textContent = money(forecastPrice);
      $("buyerForecastDelta").textContent = deltaMoney(delta);
      $("buyerForecastDelta").className = delta >= 0 ? "up" : "down";
      $("bestModel").textContent = String(prediction.model_name || "-").replace(/_/g, " ");
      $("predictionTime").textContent = `${Number(body.prediction_seconds || 0).toFixed(3)}s`;
      $("totalTime").textContent = `${Number(body.total_seconds || 0).toFixed(3)}s`;
      $("historyRows").textContent = "-";
      $("apiResponse").value = JSON.stringify(body, null, 2);

      $("buyerForecastList").innerHTML = forecast.length
        ? forecast.map((row) => `
            <div class="list-row">
              <span>Day ${row.day} <span class="muted">${row.prediction_date}</span></span>
              <strong>${money(row.predicted_price)}</strong>
            </div>
          `).join("")
        : "No forecast returned.";

      $("buyerNote").textContent = `${buyerSignal}. Day 7 change is ${deltaMoney(delta)}.`;
      renderSeller();
      renderCharts();
    }

    function renderSeller() {
      const card = selectedPokemonCard();
      if (!card) return;
      const forecast = state.pokemonPrediction?.forecast || [];
      const lastForecast = forecast.at(-1);
      const forecastPrice = Number(lastForecast?.predicted_price ?? state.pokemonPrediction?.predicted_price ?? card.market);
      const minimumPrice = Number($("minimumPrice").value || 0);
      const costPrice = Number($("costPrice").value || 0);
      const listingPrice = Math.max(Number(card.market), forecastPrice, minimumPrice);
      const profitEach = listingPrice - costPrice;
      const floorActive = forecastPrice < minimumPrice;

      $("sellerListingPrice").textContent = money(listingPrice);
      $("sellerNote").textContent = floorActive
        ? "Minimum price protects the listing from a lower forecast."
        : "Listing follows the stronger market or forecast price.";
      $("sellerMarketPrice").textContent = money(card.market);
      $("sellerMinimumPrice").textContent = money(minimumPrice);
      $("sellerProfitEach").textContent = money(profitEach);
      $("sellerProfitNote").textContent = profitEach >= 0 ? "Profitable" : "Below cost";
      $("sellerProfitNote").className = profitEach >= 0 ? "up" : "down";
      $("sellerStatus").textContent = floorActive ? "Floor active" : "Market active";
      $("sellerStatusNote").textContent = floorActive ? "Forecast below minimum" : "Market or forecast above minimum";
      $("sellerStatusNote").className = floorActive ? "warn" : "up";
      $("sellerAlert").textContent = floorActive ? "Trigger seller alert" : "No alert";
      $("sellerAlert").className = floorActive ? "warn" : "up";
      renderCharts();
    }

    function parseRecords(value) {
      const parsed = JSON.parse(value);
      const records = Array.isArray(parsed) ? parsed : parsed.history;
      if (!Array.isArray(records) || records.length === 0) {
        throw new Error("JSON must be an array or an object with a history array.");
      }
      return records;
    }

    function inferColumn(columns, candidates) {
      const lowered = new Map(columns.map((column) => [column.toLowerCase(), column]));
      for (const candidate of candidates) {
        if (lowered.has(candidate)) return lowered.get(candidate);
      }
      return columns[0];
    }

    function refreshItems() {
      const itemColumn = $("itemColumn").value;
      const items = [...new Set(state.records.map((row) => String(row[itemColumn] ?? "")).filter(Boolean))].sort();
      fillSelect($("itemSelect"), items, items[0]);
    }

    function loadRecords(records) {
      state.records = records;
      const columns = Object.keys(records[0] || {});
      const itemColumn = inferColumn(columns, ["product", "item", "name", "card", "sku"]);
      const dateColumn = inferColumn(columns, ["date", "updated_at", "created_at", "timestamp", "datetime"]);
      const priceColumn = inferColumn(columns, ["price", "market", "market_price", "current_price", "value"]);
      fillSelect($("itemColumn"), columns, itemColumn);
      fillSelect($("dateColumn"), columns, dateColumn);
      fillSelect($("priceColumn"), columns, priceColumn);
      refreshItems();
      $("mappingPanel").style.display = "block";
      showMessage(`${records.length} rows loaded. Choose an item and run custom prediction.`);
      setStatus("Data loaded");
      openScreen("screenCustom");
    }

    async function predictCustom() {
      if (!state.records.length) {
        showMessage("Load JSON data first.", "error");
        return;
      }
      const apiUrl = $("apiUrl").value.trim().replace(/\\/$/, "");
      const itemColumn = $("itemColumn").value;
      const dateColumn = $("dateColumn").value;
      const priceColumn = $("priceColumn").value;
      const item = $("itemSelect").value;
      const payload = {
        api_url: apiUrl,
        history: state.records,
        entity_col: itemColumn,
        date_col: dateColumn,
        price_col: priceColumn,
        item,
        horizon: 7
      };

      $("customPredictButton").disabled = true;
      setStatus("Training");
      showMessage("Training custom model and predicting 7 days...");
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
        state.customResult = body;
        $("bestModel").textContent = String(body.best_model || "-").replace(/_/g, " ");
        $("historyRows").textContent = String(body.history_rows || state.records.length);
        $("predictionTime").textContent = `${Number(body.prediction_seconds || 0).toFixed(3)}s`;
        $("totalTime").textContent = `${Number(body.total_seconds || 0).toFixed(3)}s`;
        $("apiResponse").value = JSON.stringify(body, null, 2);
        showMessage("Custom prediction complete.");
        setStatus("Complete");
      } catch (error) {
        showMessage(error.message, "error");
        setStatus("Error");
      } finally {
        $("customPredictButton").disabled = false;
      }
    }

    async function loadSample() {
      try {
        const response = await fetch("/sample");
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || "Could not load sample data.");
        loadRecords(body);
      } catch (error) {
        showMessage(error.message, "error");
      }
    }

    function openScreen(screenId) {
      document.querySelectorAll(".screen").forEach((screen) => screen.classList.remove("active"));
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
      $(screenId).classList.add("active");
      document.querySelector(`[data-screen="${screenId}"]`).classList.add("active");
      $("screenSubtitle").textContent = subtitles[screenId] || "";
      renderCharts();
    }

    document.addEventListener("DOMContentLoaded", () => {
      $("apiUrl").value = document.body.dataset.defaultApi || "http://127.0.0.1:8000";
      loadPokemonCards();
      $("pokemonCard").addEventListener("change", () => changePokemonCard($("pokemonCard").value));
      $("sellerPokemonCard").addEventListener("change", () => changePokemonCard($("sellerPokemonCard").value));
      $("pokemonPredictButton").addEventListener("click", predictPokemon);
      $("sellerPredictButton").addEventListener("click", predictPokemon);
      $("refreshSellerButton").addEventListener("click", renderSeller);
      $("minimumPrice").addEventListener("change", renderSeller);
      $("costPrice").addEventListener("change", renderSeller);
      $("logoutButton").addEventListener("click", logout);
      document.querySelectorAll(".role-card").forEach((button) => {
        button.addEventListener("click", () => setRole(button.dataset.role));
      });
      $("loadSampleButton").addEventListener("click", loadSample);
      $("itemColumn").addEventListener("change", refreshItems);
      $("customPredictButton").addEventListener("click", predictCustom);
      $("jsonFile").addEventListener("change", async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;
        try {
          loadRecords(parseRecords(await file.text()));
        } catch (error) {
          showMessage(error.message, "error");
        }
      });
      $("jsonText").addEventListener("change", (event) => {
        if (!event.target.value.trim()) return;
        try {
          loadRecords(parseRecords(event.target.value));
        } catch (error) {
          showMessage(error.message, "error");
        }
      });
      document.querySelectorAll(".tab").forEach((tab) => {
        tab.addEventListener("click", () => openScreen(tab.dataset.screen));
      });
    });
  </script>
</body>
</html>
"""


def normalize_api_url(value: str | None) -> str:
    return (value or API_URL).strip().rstrip("/")


def pokemon_market() -> list[dict[str, Any]]:
    df = pd.read_csv(DATA_PATH, sep=CSV_SEPARATOR)
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    df["market"] = pd.to_numeric(df["market"], errors="coerce")
    df["market_low"] = pd.to_numeric(df["market_low"], errors="coerce")
    df = df.dropna(subset=["name", "updated_at", "market", "market_low"]).sort_values(["name", "updated_at"])

    latest = df.groupby("name", sort=False).tail(1).copy()
    previous = (
        df.groupby("name", sort=False)
        .tail(2)
        .groupby("name", sort=False)
        .head(1)[["name", "market"]]
        .rename(columns={"market": "previous_market"})
    )
    latest = latest.merge(previous, on="name", how="left")
    latest["change"] = latest["market"] - latest["previous_market"]
    latest["change_pct"] = latest["change"] / latest["previous_market"] * 100
    latest["spread"] = latest["market"] - latest["market_low"]
    latest = latest.sort_values("name")

    cards = []
    for row in latest.to_dict(orient="records"):
        cards.append(
            {
                "name": row["name"],
                "id": int(row["id"]),
                "updated_at": row["updated_at"].date().isoformat(),
                "market": round(float(row["market"]), 2),
                "market_low": round(float(row["market_low"]), 2),
                "previous_market": round(float(row["previous_market"]), 2)
                if pd.notna(row["previous_market"])
                else None,
                "change": round(float(row["change"]), 2) if pd.notna(row["change"]) else 0.0,
                "change_pct": round(float(row["change_pct"]), 2) if pd.notna(row["change_pct"]) else 0.0,
                "spread": round(float(row["spread"]), 2),
            }
        )
    return cards


@app.get("/")
def index() -> str:
    return render_template_string(MOBILE_HTML, default_api_url=normalize_api_url(os.getenv("PRICING_API_URL")))


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.get("/pokemon-cards")
def pokemon_cards() -> tuple[Any, int]:
    try:
        cards = pokemon_market()
        default_card = "Alakazam" if any(card["name"] == "Alakazam" for card in cards) else cards[0]["name"]
        return jsonify({"cards": cards, "default_card": default_card}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/sample")
def sample() -> tuple[Any, int]:
    try:
        return jsonify(json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict")
def predict() -> tuple[Any, int]:
    try:
        payload = request.get_json(silent=True) or {}
        api_url = normalize_api_url(payload.pop("api_url", None))
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=180)
            try:
                body = response.json()
            except ValueError:
                body = {"error": response.text}
            return jsonify(body), response.status_code
        except requests.RequestException:
            if "history" in payload:
                raise

            df = load_pricing_data(DATA_PATH, DATE_COLUMN, PRICE_COLUMN, CSV_SEPARATOR)
            horizon = int(payload.get("horizon", 7))
            predictions = predict_future_prices(df, MODEL_DIR, payload.get("item"), horizon=horizon)
            return jsonify(
                {
                    "mode": "pokemon_local_fallback",
                    "predictions": predictions,
                    "forecast_days": horizon,
                    "prediction_seconds": 0,
                    "total_seconds": 0,
                }
            ), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(
        host=os.getenv("MOBILE_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8600")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
