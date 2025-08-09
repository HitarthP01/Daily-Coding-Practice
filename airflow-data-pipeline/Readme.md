# Airflow Data Pipeline Project

Welcome to the **Airflow Data Pipeline** project! This repository contains a Dockerized Apache Airflow setup integrated with Redis, designed to process and analyze sample data using a simple ETL (Extract, Transform, Load) workflow. This project was created as part of the `Daily-Coding-Practice` initiative to explore data pipeline automation and orchestration.

## Purpose
The primary goal of this project is to:
- Demonstrate a basic ETL pipeline using Apache Airflow.
- Learn and implement containerization with Docker for reproducibility.
- Experiment with Redis as a task queue executor to scale Airflow workflows.
- Serve as a reusable template for future data pipeline projects.

This setup reads data from a CSV file, calculates the average price, and stores the result in a SQLite database, all orchestrated by Airflow with Redis handling task scheduling.

## Prerequisites
- **Docker**: Ensure Docker Desktop is installed and running (version 20.10+ recommended).
- **Git**: Installed for version control (version 2.30+ recommended).
- **Operating System**: Works on Windows, macOS, or Linux.
- **Internet Connection**: Required for pulling Docker images.

## Project Structure
