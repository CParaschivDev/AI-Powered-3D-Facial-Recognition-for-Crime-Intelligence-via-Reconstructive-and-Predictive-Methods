@echo off
echo Installing required packages for the frontend...
echo.

cd "%~dp0"
echo Installing charting and utility dependencies...
npm install react-chartjs-2 chart.js html2canvas --save

echo Installing @craco/craco for webpack configuration overrides...
npm install @craco/craco --save

echo.
echo Done! Now you can start the application with: npm start