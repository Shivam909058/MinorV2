FROM node:16

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm install

COPY . .

RUN npm run build

# Serve the app using serve
RUN npm install -g serve

CMD ["serve", "-s", "build"]