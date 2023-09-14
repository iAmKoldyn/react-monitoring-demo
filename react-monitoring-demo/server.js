const express = require('express');
const promClient = require('prom-client');
const path = require('path');

const app = express();
const port = 3001;

const register = promClient.register;
promClient.collectDefaultMetrics();

const pageViews = new promClient.Counter({
    name: 'page_views',
    help: 'Number of page views'
});

app.get('/', (req, res) => {
    pageViews.inc();
    res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.get('/metrics', async (req, res) => {
    res.set('Content-Type', register.contentType);
    try {
        const metrics = await register.metrics();
        res.end(metrics);
    } catch (err) {
        res.status(500).end(err);
    }
});

app.use(express.static(path.join(__dirname, 'build')));

app.listen(port, () => {
    console.log(`Server :${port}`);
});
