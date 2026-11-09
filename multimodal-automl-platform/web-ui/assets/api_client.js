window.AutoMLApiClient = {
    gatewayUrl: "http://127.0.0.1:3000/api/gateway/submit",
    predictUrl: "http://127.0.0.1:3000/api/gateway/predict",

    async submitToGateway(payload) {
        try {
            const res = await fetch(this.gatewayUrl, {
                method: "POST", headers: { "Content-Type": "application/json", "x-api-key": "local-dev-key" },
                body: JSON.stringify(payload)
            });
            return await res.json();
        } catch (err) {
            return window.StageTester.runBrowserFallbackValidationAndExecution(payload);
        }
    },
    
    async predict(payload) {
        try {
            const res = await fetch(this.predictUrl, {
                method: "POST", headers: { "Content-Type": "application/json", "x-api-key": "local-dev-key" },
                body: JSON.stringify(payload)
            });
            return await res.json();
        } catch (err) {
            return window.StageTester.runBrowserFallbackPrediction(payload);
        }
    }
};
