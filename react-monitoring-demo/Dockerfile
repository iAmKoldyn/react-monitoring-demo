FROM node:14

WORKDIR /app

COPY package*.json ./

RUN npm install
RUN npm install prom-client@13.1.0 --save

COPY . .

CMD ["node", "server.js"]
