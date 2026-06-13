"""
portfolio/loader.py
-------------------
Funciones para cargar los datos de un portfolio: datos históricos de futuros
y artefactos de modelos (scaler, features, pesos, params). Diseñado para
aprovechar utilidades existentes en `tradepy.data.loader` y `tradepy.load`.

API principal:
	load_portfolio_assets(tickers, load_models=True, model_classes=None, device='cpu')

El resultado es un dict por ticker con las claves `data`, `spec`, `models`.
Cada entrada en `models` contiene los artefactos cargados y, opcionalmente,
la instancia del modelo si se proporciona la clase en `model_classes`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import joblib

from tradepy.data.loader import load_future
from tradepy.paths import MODELS_DIR
from tradepy.load import load_trading_model, load_trading_model_2level

log = logging.getLogger(__name__)


def _find_model_base_dir_for_ticker(ticker: str) -> Optional[Path]:
	"""Busca la carpeta base de modelos para un ticker bajo `MODELS_DIR`.

	Intenta varios matches case-insensitive y devuelve la primera carpeta
	encontrada o `None` si no existe.
	"""
	ticker_lower = ticker.lower()
	# 1) Ruta directa: MODELS_DIR / ticker
	p1 = MODELS_DIR / ticker_lower
	if p1.exists() and p1.is_dir():
		return p1

	# 2) Ruta con mayúsculas
	p2 = MODELS_DIR / ticker.upper()
	if p2.exists() and p2.is_dir():
		return p2

	# 3) Buscar entre hijos del directorio MODELS_DIR
	if MODELS_DIR.exists():
		for child in MODELS_DIR.iterdir():
			if child.is_dir() and child.name.lower() == ticker_lower:
				return child

	return None


def _list_model_dirs(base_dir: Path) -> List[Path]:
	"""Lista subdirectorios (cada uno representando un artefacto/versión de modelo)."""
	return [p for p in base_dir.iterdir() if p.is_dir()]


def _load_model_artifacts(model_dir: Path) -> Dict[str, Any]:
	"""Carga artefactos no específicos de framework desde una carpeta de modelo.

	Devuelve un dict con keys: model_type ("single"|"2level"|"keras"|None),
	model_params, features, scaler, weights (path o dict), best_thresholds.
	"""
	res: Dict[str, Any] = {
		"path": model_dir,
		"model_type": None,
		"model_params": None,
		"features": None,
		"scaler": None,
		"weights": None,
		"best_thresholds": None,
	}

	try:
		if (model_dir / "model_params.pkl").exists():
			res["model_params"] = joblib.load(model_dir / "model_params.pkl")
		if (model_dir / "features.pkl").exists():
			res["features"] = joblib.load(model_dir / "features.pkl")
		if (model_dir / "scaler.pkl").exists():
			res["scaler"] = joblib.load(model_dir / "scaler.pkl")
		if (model_dir / "best_thresholds.pkl").exists():
			res["best_thresholds"] = joblib.load(model_dir / "best_thresholds.pkl")

		# Pesos / tipo de modelo
		if (model_dir / "model_weights.pth").exists():
			res["model_type"] = "single"
			res["weights"] = model_dir / "model_weights.pth"
		elif (model_dir / "model_l1_weights.pth").exists() and (model_dir / "model_l2_weights.pth").exists():
			res["model_type"] = "2level"
			res["weights"] = {
				"l1": model_dir / "model_l1_weights.pth",
				"l2": model_dir / "model_l2_weights.pth",
			}
		else:
			# Buscar artefactos keras (.h5/.hdf5)
			h5 = list(model_dir.glob("*.h5")) + list(model_dir.glob("*.hdf5"))
			if h5:
				res["model_type"] = "keras"
				res["weights"] = h5[0]
	except Exception as e:  # pragma: no cover - robust against corrupt files
		log.exception("Error cargando artefactos en %s: %s", model_dir, e)

	return res


def load_portfolio_assets(
	tickers: List[str],
	load_models: bool = True,
	model_classes: Optional[Dict[str, Any]] = None,
	device: str = "cpu",
) -> Dict[str, Any]:
	"""Carga datos + artefactos de modelos para una lista de futuros.

	Parámetros
	----------
	tickers:
		Lista de tickers (ej. ['GC', 'ES']).
	load_models:
		Si True intentará instanciar modelos cuando se proporcione `model_classes`.
	model_classes:
		Dict opcional que mapea ticker -> ModelClass ó (ModelClassL1, ModelClassL2)
		para pipelines de 2 niveles.
	device:
		Dispositivo para cargar pesos torch ('cpu' o 'cuda').

	Retorna
	-------
	dict: { ticker: { 'data': DataFrame|None, 'spec': dict|None, 'models': { model_dir_name: {artifacts, instance, load_error}} , 'errors': [] } }
	"""
	results: Dict[str, Any] = {}

	for t in tickers:
		ticker = t.upper()
		entry: Dict[str, Any] = {"data": None, "spec": None, "models": {}, "errors": []}

		# 1) Cargar datos y spec usando utilidades existentes
		try:
			df, spec = load_future(ticker)
			entry["data"] = df
			entry["spec"] = spec
		except Exception as e:
			log.exception("Error cargando datos para %s", ticker)
			entry["errors"].append(f"data_load_error: {e}")
			results[ticker] = entry
			continue

		# 2) Localizar carpeta de modelos
		base_dir = _find_model_base_dir_for_ticker(ticker)
		if base_dir is None:
			entry["errors"].append("models_dir_not_found")
			results[ticker] = entry
			continue

		model_dirs = _list_model_dirs(base_dir)
		if not model_dirs:
			entry["errors"].append("no_model_subdirs")
			results[ticker] = entry
			continue

		# 3) Para cada versión/artefacto de modelo, cargar artefactos y opcionalmente instanciar
		for md in sorted(model_dirs, key=lambda p: p.name):
			art = _load_model_artifacts(md)
			model_info: Dict[str, Any] = {"artifacts": art, "instance": None, "load_error": None}

			if load_models:
				try:
					if model_classes and ticker in model_classes:
						mc = model_classes[ticker]
						if art["model_type"] == "2level":
							if isinstance(mc, (list, tuple)) and len(mc) >= 2:
								model_l1, model_l2, scaler, features = load_trading_model_2level(
									mc[0], mc[1], path=str(md), device=device
								)
								model_info["instance"] = {"l1": model_l1, "l2": model_l2}
								model_info["artifacts"]["scaler"] = scaler
								model_info["artifacts"]["features"] = features
							else:
								raise ValueError("model_classes para modelos 2level debe ser (class_l1, class_l2)")
						elif art["model_type"] == "single":
							model, scaler, features = load_trading_model(mc, path=str(md), device=device)
							model_info["instance"] = model
							model_info["artifacts"]["scaler"] = scaler
							model_info["artifacts"]["features"] = features
						else:
							# Intento por defecto con load_trading_model si se provee una clase
							if isinstance(mc, (list, tuple)):
								model, scaler, features = load_trading_model(mc[0], path=str(md), device=device)
								model_info["instance"] = model
								model_info["artifacts"]["scaler"] = scaler
								model_info["artifacts"]["features"] = features
							else:
								model, scaler, features = load_trading_model(mc, path=str(md), device=device)
								model_info["instance"] = model
								model_info["artifacts"]["scaler"] = scaler
								model_info["artifacts"]["features"] = features
					else:
						# No se proporcionó clase de modelo: mantenemos solo los artefactos cargados
						pass
				except Exception as e:  # pragma: no cover - registrar error de carga de modelo
					log.exception("Error instanciando modelo %s/%s: %s", ticker, md, e)
					model_info["load_error"] = str(e)

			entry["models"][md.name] = model_info

		results[ticker] = entry

	return results


__all__ = ["load_portfolio_assets"]

