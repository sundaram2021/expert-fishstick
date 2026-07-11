.PHONY: up down install build test loadtest clean

## Start the full system (gateway + model backend) in one command
up:
	docker compose up --build

down:
	docker compose down -v

## Local development without Docker
install:
	cd services/model-backend && npm install
	cd services/gateway && npm install

build:
	cd services/model-backend && npm run build
	cd services/gateway && npm run build

test:
	cd services/model-backend && npm test
	cd services/gateway && npm test

## Run the phased load test against a running system (docker or local)
loadtest:
	node loadtest/run.mjs

clean:
	rm -rf services/gateway/dist services/model-backend/dist
