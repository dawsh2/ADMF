# Reporting System

## Overview

The Reporting System provides a comprehensive framework for generating, scheduling, and distributing reports that give insights into trading strategy performance, system status, and operational metrics. It transforms raw data and metrics into actionable information for users across different roles.

## Problem Statement

Trading systems generate vast amounts of data that require transformation into structured insights:

1. **Data Synthesis**: Raw trading data, performance metrics, and system events need to be transformed into meaningful reports

2. **Audience-Specific Reports**: Different users (traders, system administrators, risk managers) need different views of the data

3. **Multiple Time Horizons**: Reports need to cover various time periods (intraday, daily, weekly, monthly)

4. **Delivery Mechanisms**: Reports must be available through multiple channels (web dashboards, PDF documents, email)

5. **Data Volume**: Large data volumes require efficient processing and summarization techniques

## Design Solution

### 1. Core Reporting Engine

The foundation of the reporting system is a flexible, template-based engine:

```python
import os
import json
import datetime
import logging
import threading
import jinja2
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Callable, Type

class ReportFormat(Enum):
    """Report output formats."""
    HTML = auto()
    PDF = auto()
    JSON = auto()
    CSV = auto()
    EXCEL = auto()
    TEXT = auto()
    
class ReportPeriod(Enum):
    """Report time periods."""
    INTRADAY = auto()
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    QUARTERLY = auto()
    YEARLY = auto()
    CUSTOM = auto()
    
class ReportType(Enum):
    """Report types."""
    STRATEGY_PERFORMANCE = auto()
    SYSTEM_STATUS = auto()
    OPERATIONAL = auto()
    RISK = auto()
    CUSTOM = auto()

class ReportTemplate:
    """Base class for report templates."""
    
    def __init__(self, name, description=None, template_path=None, template_string=None):
        """
        Initialize report template.
        
        Args:
            name: Template name
            description: Optional description
            template_path: Path to template file
            template_string: Template string content
        """
        self.name = name
        self.description = description
        self.template_path = template_path
        self.template_string = template_string
        
        # Ensure we have either path or string
        if template_path is None and template_string is None:
            raise ValueError("Either template_path or template_string must be provided")
            
        # Initialize Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.dirname(template_path)) if template_path else jinja2.DictLoader({}),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Load template
        if template_path:
            self.template = self.env.get_template(os.path.basename(template_path))
        else:
            self.template = self.env.from_string(template_string)
            
    def render(self, data):
        """
        Render template with data.
        
        Args:
            data: Data for template rendering
            
        Returns:
            Rendered template
        """
        return self.template.render(**data)

class ReportDefinition:
    """Definition of a report type."""
    
    def __init__(self, name, description=None, report_type=ReportType.CUSTOM, 
               template=None, format=ReportFormat.HTML, period=ReportPeriod.DAILY):
        """
        Initialize report definition.
        
        Args:
            name: Report name
            description: Optional description
            report_type: Report type
            template: Report template
            format: Report output format
            period: Report time period
        """
        self.name = name
        self.description = description
        self.report_type = report_type
        self.template = template
        self.format = format
        self.period = period
        self.data_providers = []
        self.post_processors = []
        
    def add_data_provider(self, provider):
        """
        Add data provider to report.
        
        Args:
            provider: Data provider function or object
            
        Returns:
            Self for chaining
        """
        self.data_providers.append(provider)
        return self
        
    def add_post_processor(self, processor):
        """
        Add post-processor to report.
        
        Args:
            processor: Post-processor function or object
            
        Returns:
            Self for chaining
        """
        self.post_processors.append(processor)
        return self
        
    def collect_data(self, parameters=None):
        """
        Collect data from all providers.
        
        Args:
            parameters: Optional parameters for data collection
            
        Returns:
            Collected data
        """
        data = {}
        parameters = parameters or {}
        
        # Collect data from all providers
        for provider in self.data_providers:
            if callable(provider):
                provider_data = provider(parameters)
            elif hasattr(provider, 'get_data'):
                provider_data = provider.get_data(parameters)
            else:
                raise TypeError("Data provider must be callable or have get_data method")
                
            # Merge with existing data
            data.update(provider_data)
            
        return data
        
    def process_data(self, data, parameters=None):
        """
        Apply post-processors to data.
        
        Args:
            data: Data to process
            parameters: Optional parameters for processing
            
        Returns:
            Processed data
        """
        processed_data = data.copy()
        parameters = parameters or {}
        
        # Apply all post-processors
        for processor in self.post_processors:
            if callable(processor):
                processed_data = processor(processed_data, parameters)
            elif hasattr(processor, 'process'):
                processed_data = processor.process(processed_data, parameters)
            else:
                raise TypeError("Post-processor must be callable or have process method")
                
        return processed_data
        
    def generate_report(self, parameters=None):
        """
        Generate report.
        
        Args:
            parameters: Optional parameters for report generation
            
        Returns:
            Generated report content
        """
        parameters = parameters or {}
        
        # Collect raw data
        raw_data = self.collect_data(parameters)
        
        # Process data
        processed_data = self.process_data(raw_data, parameters)
        
        # Add metadata
        metadata = {
            'report_name': self.name,
            'report_type': self.report_type.name,
            'generated_at': datetime.datetime.now(),
            'parameters': parameters
        }
        
        processed_data['metadata'] = metadata
        
        # Render template
        if self.template:
            if self.format == ReportFormat.HTML:
                return self.template.render(processed_data)
            else:
                # For non-HTML formats, render template first then convert
                html_content = self.template.render(processed_data)
                return self._convert_format(html_content, processed_data)
        else:
            # No template, return raw data in requested format
            return self._format_data(processed_data)
            
    def _convert_format(self, html_content, data):
        """
        Convert HTML to target format.
        
        Args:
            html_content: HTML content
            data: Report data
            
        Returns:
            Converted content
        """
        if self.format == ReportFormat.PDF:
            return self._html_to_pdf(html_content)
        elif self.format == ReportFormat.JSON:
            return json.dumps(data, default=str, indent=2)
        elif self.format == ReportFormat.CSV:
            return self._data_to_csv(data)
        elif self.format == ReportFormat.EXCEL:
            return self._data_to_excel(data)
        elif self.format == ReportFormat.TEXT:
            return self._html_to_text(html_content)
        else:
            return html_content
            
    def _html_to_pdf(self, html_content):
        """Convert HTML to PDF."""
        try:
            import weasyprint
            return weasyprint.HTML(string=html_content).write_pdf()
        except ImportError:
            logging.warning("WeasyPrint not installed. Falling back to HTML.")
            return html_content
            
    def _html_to_text(self, html_content):
        """Convert HTML to text."""
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(html_content, 'html.parser').get_text()
        except ImportError:
            logging.warning("BeautifulSoup not installed. Returning HTML.")
            return html_content
            
    def _data_to_csv(self, data):
        """Convert data to CSV."""
        # Find dataframes in data
        csv_parts = []
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                csv_parts.append(f"# {key}")
                csv_parts.append(value.to_csv())
                
        return "\n\n".join(csv_parts)
        
    def _data_to_excel(self, data):
        """Convert data to Excel."""
        # Find dataframes in data
        try:
            import io
            output = io.BytesIO()
            
            with pd.ExcelWriter(output) as writer:
                for key, value in data.items():
                    if isinstance(value, pd.DataFrame):
                        value.to_excel(writer, sheet_name=key[:31])  # Excel limits sheet names to 31 chars
                        
            return output.getvalue()
        except ImportError:
            logging.warning("Pandas ExcelWriter not available. Falling back to CSV.")
            return self._data_to_csv(data)
            
    def _format_data(self, data):
        """Format data according to target format."""
        if self.format == ReportFormat.JSON:
            return json.dumps(data, default=str, indent=2)
        elif self.format == ReportFormat.CSV:
            return self._data_to_csv(data)
        elif self.format == ReportFormat.EXCEL:
            return self._data_to_excel(data)
        elif self.format == ReportFormat.TEXT:
            return str(data)
        else:
            return data

class ReportGenerator:
    """Generator for reports."""
    
    def __init__(self):
        """Initialize report generator."""
        self.report_definitions = {}
        self.templates = {}
        
    def register_template(self, template):
        """
        Register report template.
        
        Args:
            template: Report template
            
        Returns:
            Registered template
        """
        self.templates[template.name] = template
        return template
        
    def register_report(self, report_definition):
        """
        Register report definition.
        
        Args:
            report_definition: Report definition
            
        Returns:
            Registered report definition
        """
        self.report_definitions[report_definition.name] = report_definition
        return report_definition
        
    def create_report(self, name, parameters=None):
        """
        Create report by name.
        
        Args:
            name: Report name
            parameters: Optional parameters
            
        Returns:
            Generated report content
        """
        if name not in self.report_definitions:
            raise ValueError(f"Report definition '{name}' not found")
            
        # Get report definition
        report_def = self.report_definitions[name]
        
        # Generate report
        return report_def.generate_report(parameters)
        
    def get_report_definitions(self):
        """
        Get all report definitions.
        
        Returns:
            Dictionary of report definitions
        """
        return self.report_definitions.copy()
        
    def get_templates(self):
        """
        Get all report templates.
        
        Returns:
            Dictionary of report templates
        """
        return self.templates.copy()
```

### 2. Report Scheduling System

Automated report scheduling and delivery mechanism:

```python
import schedule
import time
import threading
import logging
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Dict, List, Any, Optional, Union, Callable, Type

class ReportDeliveryMethod(Enum):
    """Report delivery methods."""
    EMAIL = auto()
    FILE = auto()
    HTTP = auto()
    CONSOLE = auto()
    CUSTOM = auto()

class ReportSchedule:
    """Schedule for report generation."""
    
    def __init__(self, report_name, schedule_type, parameters=None,
               delivery_method=ReportDeliveryMethod.FILE,
               delivery_parameters=None,
               enabled=True):
        """
        Initialize report schedule.
        
        Args:
            report_name: Name of report to generate
            schedule_type: Schedule type (daily, weekly, etc.)
            parameters: Optional parameters for report generation
            delivery_method: Method for delivering report
            delivery_parameters: Parameters for delivery method
            enabled: Whether schedule is enabled
        """
        self.report_name = report_name
        self.schedule_type = schedule_type
        self.parameters = parameters or {}
        self.delivery_method = delivery_method
        self.delivery_parameters = delivery_parameters or {}
        self.enabled = enabled
        self.last_run = None
        self.next_run = None
        self.job = None
        
    def as_dict(self):
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'report_name': self.report_name,
            'schedule_type': self.schedule_type,
            'parameters': self.parameters,
            'delivery_method': self.delivery_method.name,
            'delivery_parameters': self.delivery_parameters,
            'enabled': self.enabled,
            'last_run': self.last_run,
            'next_run': self.next_run
        }

class ReportScheduler:
    """Scheduler for report generation."""
    
    def __init__(self, report_generator):
        """
        Initialize report scheduler.
        
        Args:
            report_generator: ReportGenerator instance
        """
        self.report_generator = report_generator
        self.schedules = {}
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        
    def add_schedule(self, report_name, schedule_type, parameters=None,
                    delivery_method=ReportDeliveryMethod.FILE,
                    delivery_parameters=None,
                    enabled=True):
        """
        Add report schedule.
        
        Args:
            report_name: Name of report to generate
            schedule_type: Schedule type (daily, weekly, etc.)
            parameters: Optional parameters for report generation
            delivery_method: Method for delivering report
            delivery_parameters: Parameters for delivery method
            enabled: Whether schedule is enabled
            
        Returns:
            Created schedule
        """
        with self.lock:
            # Create schedule object
            schedule_obj = ReportSchedule(
                report_name=report_name,
                schedule_type=schedule_type,
                parameters=parameters,
                delivery_method=delivery_method,
                delivery_parameters=delivery_parameters,
                enabled=enabled
            )
            
            # Generate schedule ID
            schedule_id = f"{report_name}_{len(self.schedules)}"
            
            # Store schedule
            self.schedules[schedule_id] = schedule_obj
            
            # Schedule job if enabled
            if enabled:
                self._schedule_job(schedule_id, schedule_obj)
                
            return schedule_id
            
    def _schedule_job(self, schedule_id, schedule_obj):
        """
        Schedule job based on schedule type.
        
        Args:
            schedule_id: Schedule identifier
            schedule_obj: Schedule object
            
        Returns:
            Scheduled job
        """
        # Define job function
        def job_func():
            self._run_report(schedule_id)
            
        # Schedule based on type
        schedule_type = schedule_obj.schedule_type
        job = None
        
        if schedule_type == 'daily':
            # Extract time parameter or use default
            time_param = schedule_obj.parameters.get('time', '00:00')
            job = schedule.every().day.at(time_param).do(job_func)
            
        elif schedule_type == 'weekly':
            # Extract day and time parameters
            day_param = schedule_obj.parameters.get('day', 'monday')
            time_param = schedule_obj.parameters.get('time', '00:00')
            
            # Get day method
            day_method = getattr(schedule.every(), day_param.lower())
            job = day_method.at(time_param).do(job_func)
            
        elif schedule_type == 'monthly':
            # Extract day of month and time parameters
            day_param = schedule_obj.parameters.get('day', 1)
            time_param = schedule_obj.parameters.get('time', '00:00')
            
            # Schedule monthly job
            job = schedule.every().month.do(job_func)
            
        elif schedule_type == 'hourly':
            # Extract minute parameter
            minute_param = schedule_obj.parameters.get('minute', 0)
            job = schedule.every().hour.at(f":{minute_param:02d}").do(job_func)
            
        elif schedule_type == 'interval':
            # Extract interval parameters
            interval = schedule_obj.parameters.get('interval', 60)  # seconds
            job = schedule.every(interval).seconds.do(job_func)
            
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        # Store job in schedule object
        schedule_obj.job = job
        
        # Calculate next run
        if job:
            schedule_obj.next_run = job.next_run
            
        return job
        
    def _run_report(self, schedule_id):
        """
        Run report for given schedule.
        
        Args:
            schedule_id: Schedule identifier
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if schedule_id not in self.schedules:
                logging.error(f"Schedule {schedule_id} not found")
                return False
                
            schedule_obj = self.schedules[schedule_id]
            
            try:
                # Generate report
                report_content = self.report_generator.create_report(
                    schedule_obj.report_name,
                    schedule_obj.parameters
                )
                
                # Deliver report
                self._deliver_report(report_content, schedule_obj)
                
                # Update schedule metadata
                schedule_obj.last_run = datetime.datetime.now()
                if schedule_obj.job:
                    schedule_obj.next_run = schedule_obj.job.next_run
                    
                return True
            except Exception as e:
                logging.error(f"Error running report {schedule_obj.report_name}: {e}")
                return False
                
    def _deliver_report(self, report_content, schedule_obj):
        """
        Deliver report according to delivery method.
        
        Args:
            report_content: Report content
            schedule_obj: Schedule object
            
        Returns:
            True if successful, False otherwise
        """
        method = schedule_obj.delivery_method
        params = schedule_obj.delivery_parameters
        
        if method == ReportDeliveryMethod.EMAIL:
            return self._deliver_email(report_content, params, schedule_obj)
            
        elif method == ReportDeliveryMethod.FILE:
            return self._deliver_file(report_content, params, schedule_obj)
            
        elif method == ReportDeliveryMethod.HTTP:
            return self._deliver_http(report_content, params, schedule_obj)
            
        elif method == ReportDeliveryMethod.CONSOLE:
            return self._deliver_console(report_content, params, schedule_obj)
            
        elif method == ReportDeliveryMethod.CUSTOM:
            if 'delivery_func' in params and callable(params['delivery_func']):
                return params['delivery_func'](report_content, schedule_obj)
            else:
                logging.error("Custom delivery method requires delivery_func parameter")
                return False
                
        else:
            logging.error(f"Unknown delivery method: {method}")
            return False
            
    def _deliver_email(self, report_content, params, schedule_obj):
        """Deliver report via email."""
        # Extract parameters
        recipients = params.get('recipients', [])
        subject = params.get('subject', f"Report: {schedule_obj.report_name}")
        sender = params.get('sender', 'reports@example.com')
        smtp_server = params.get('smtp_server', 'localhost')
        smtp_port = params.get('smtp_port', 25)
        smtp_user = params.get('smtp_user')
        smtp_password = params.get('smtp_password')
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject
        
        # Determine content type
        report_def = self.report_generator.report_definitions.get(schedule_obj.report_name)
        
        if report_def and report_def.format == ReportFormat.HTML:
            # HTML report
            msg.attach(MIMEText(report_content, 'html'))
        else:
            # Handle binary content
            if isinstance(report_content, bytes):
                filename = f"{schedule_obj.report_name}.{report_def.format.name.lower()}"
                attachment = MIMEApplication(report_content)
                attachment['Content-Disposition'] = f'attachment; filename="{filename}"'
                msg.attach(attachment)
                
                # Add plain text body
                msg.attach(MIMEText(f"Please see attached report: {filename}", 'plain'))
            else:
                # Plain text content
                msg.attach(MIMEText(report_content, 'plain'))
                
        # Send email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
            return True
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            return False
            
    def _deliver_file(self, report_content, params, schedule_obj):
        """Deliver report to file."""
        # Extract parameters
        directory = params.get('directory', 'reports')
        filename_template = params.get('filename', '{report_name}_{date}.{ext}')
        
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Determine file extension
        report_def = self.report_generator.report_definitions.get(schedule_obj.report_name)
        extension = report_def.format.name.lower() if report_def else 'txt'
        
        # Format filename
        date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = filename_template.format(
            report_name=schedule_obj.report_name,
            date=date_str,
            ext=extension
        )
        
        # Write to file
        file_path = os.path.join(directory, filename)
        try:
            if isinstance(report_content, bytes):
                with open(file_path, 'wb') as f:
                    f.write(report_content)
            else:
                with open(file_path, 'w') as f:
                    f.write(report_content)
            return True
        except Exception as e:
            logging.error(f"Error writing to file {file_path}: {e}")
            return False
            
    def _deliver_http(self, report_content, params, schedule_obj):
        """Deliver report via HTTP."""
        # Extract parameters
        url = params.get('url')
        method = params.get('method', 'POST')
        headers = params.get('headers', {})
        
        if not url:
            logging.error("HTTP delivery requires url parameter")
            return False
            
        # Send HTTP request
        try:
            import requests
            
            # Determine content type
            report_def = self.report_generator.report_definitions.get(schedule_obj.report_name)
            
            if isinstance(report_content, bytes):
                files = {'report': (f"{schedule_obj.report_name}.{report_def.format.name.lower()}", report_content)}
                response = requests.request(method, url, headers=headers, files=files)
            else:
                response = requests.request(method, url, headers=headers, data=report_content)
                
            # Check response
            response.raise_for_status()
            return True
        except ImportError:
            logging.error("Requests library not installed")
            return False
        except Exception as e:
            logging.error(f"Error sending HTTP request: {e}")
            return False
            
    def _deliver_console(self, report_content, params, schedule_obj):
        """Deliver report to console."""
        print("-" * 80)
        print(f"Report: {schedule_obj.report_name}")
        print(f"Generated: {datetime.datetime.now()}")
        print("-" * 80)
        
        if isinstance(report_content, bytes):
            print(f"Binary content of size {len(report_content)} bytes")
        else:
            print(report_content)
            
        print("-" * 80)
        return True
        
    def enable_schedule(self, schedule_id):
        """
        Enable report schedule.
        
        Args:
            schedule_id: Schedule identifier
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if schedule_id not in self.schedules:
                return False
                
            schedule_obj = self.schedules[schedule_id]
            
            if not schedule_obj.enabled:
                schedule_obj.enabled = True
                
                # Reschedule job
                self._schedule_job(schedule_id, schedule_obj)
                
            return True
            
    def disable_schedule(self, schedule_id):
        """
        Disable report schedule.
        
        Args:
            schedule_id: Schedule identifier
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if schedule_id not in self.schedules:
                return False
                
            schedule_obj = self.schedules[schedule_id]
            
            if schedule_obj.enabled:
                schedule_obj.enabled = False
                
                # Cancel scheduled job
                if schedule_obj.job:
                    schedule.cancel_job(schedule_obj.job)
                    schedule_obj.job = None
                    
            return True
            
    def get_schedules(self):
        """
        Get all report schedules.
        
        Returns:
            Dictionary of schedule objects
        """
        with self.lock:
            return {sid: schedule_obj.as_dict() for sid, schedule_obj in self.schedules.items()}
            
    def get_schedule(self, schedule_id):
        """
        Get report schedule by ID.
        
        Args:
            schedule_id: Schedule identifier
            
        Returns:
            Schedule object or None
        """
        with self.lock:
            schedule_obj = self.schedules.get(schedule_id)
            return schedule_obj.as_dict() if schedule_obj else None
            
    def start(self):
        """Start scheduler thread."""
        with self.lock:
            if self.running:
                return
                
            self.running = True
            
            # Create thread for scheduler
            self.thread = threading.Thread(target=self._scheduler_thread, daemon=True)
            self.thread.start()
            
    def stop(self):
        """Stop scheduler thread."""
        with self.lock:
            if not self.running:
                return
                
            self.running = False
            
            # Wait for thread to terminate
            if self.thread:
                self.thread.join(timeout=1.0)
                self.thread = None
                
    def _scheduler_thread(self):
        """Scheduler thread function."""
        while self.running:
            try:
                # Run pending jobs
                schedule.run_pending()
                
                # Sleep briefly
                time.sleep(1.0)
            except Exception as e:
                logging.error(f"Error in scheduler thread: {e}")
                time.sleep(5.0)  # Sleep longer on error
```

### 3. Domain-Specific Reports

Specialized reports for different domains:

```python
### Strategy Performance Reports

class StrategyPerformanceDataProvider:
    """Data provider for strategy performance reports."""
    
    def __init__(self, strategy_metrics_service):
        """
        Initialize data provider.
        
        Args:
            strategy_metrics_service: Service for accessing strategy metrics
        """
        self.metrics_service = strategy_metrics_service
        
    def get_data(self, parameters):
        """
        Get strategy performance data.
        
        Args:
            parameters: Parameters for data retrieval
            
        Returns:
            Strategy performance data
        """
        # Extract parameters
        strategy_id = parameters.get('strategy_id')
        start_date = parameters.get('start_date')
        end_date = parameters.get('end_date')
        
        if not strategy_id:
            raise ValueError("strategy_id parameter is required")
            
        # Get strategy metrics
        metrics = self.metrics_service.get_strategy_metrics(strategy_id)
        
        if not metrics:
            return {'error': f"Strategy metrics not found for {strategy_id}"}
            
        # Get strategy returns
        returns = self.metrics_service.get_strategy_returns(
            strategy_id, 
            start_date=start_date,
            end_date=end_date
        )
        
        # Get equity curve
        equity_curve = self.metrics_service.get_equity_curve(
            strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get trades
        trades = self.metrics_service.get_trades(
            strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate summary metrics
        summary = self._calculate_summary(returns, equity_curve, trades)
        
        # Combine data
        return {
            'strategy': {
                'id': strategy_id,
                'name': metrics.get('name', strategy_id),
                'description': metrics.get('description', '')
            },
            'summary': summary,
            'returns': returns,
            'equity_curve': equity_curve,
            'trades': trades
        }
        
    def _calculate_summary(self, returns, equity_curve, trades):
        """Calculate summary metrics."""
        import numpy as np
        
        # Initialize summary
        summary = {}
        
        # Calculate return metrics
        if len(returns) > 0:
            # Convert to numpy array for calculations
            returns_array = np.array(returns['return'])
            
            # Calculate return statistics
            summary['total_return'] = ((returns_array + 1).prod() - 1) * 100
            summary['annualized_return'] = ((returns_array + 1).prod() ** (252 / len(returns_array)) - 1) * 100
            summary['volatility'] = returns_array.std() * np.sqrt(252) * 100
            summary['sharpe_ratio'] = summary['annualized_return'] / summary['volatility'] if summary['volatility'] > 0 else 0
            
            # Calculate drawdown
            if len(equity_curve) > 0:
                equity_values = equity_curve['equity']
                max_drawdown = 0
                peak = equity_values[0]
                
                for value in equity_values:
                    if value > peak:
                        peak = value
                    else:
                        drawdown = (peak - value) / peak
                        max_drawdown = max(max_drawdown, drawdown)
                        
                summary['max_drawdown'] = max_drawdown * 100
                
        # Calculate trade metrics
        if len(trades) > 0:
            summary['trade_count'] = len(trades)
            
            # Calculate win rate
            win_count = sum(1 for trade in trades['trades'] if trade['return'] > 0)
            summary['win_rate'] = (win_count / len(trades)) * 100
            
            # Calculate profit factor
            gross_profit = sum(trade['return'] for trade in trades['trades'] if trade['return'] > 0)
            gross_loss = abs(sum(trade['return'] for trade in trades['trades'] if trade['return'] < 0))
            summary['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate average trade metrics
            summary['avg_trade_return'] = trades['trades'].mean()
            summary['avg_win_return'] = np.mean([trade['return'] for trade in trades['trades'] if trade['return'] > 0])
            summary['avg_loss_return'] = np.mean([trade['return'] for trade in trades['trades'] if trade['return'] < 0])
            
        return summary

class StrategyPerformanceChartGenerator:
    """Chart generator for strategy performance reports."""
    
    def process(self, data, parameters):
        """
        Generate charts for strategy performance data.
        
        Args:
            data: Strategy performance data
            parameters: Processing parameters
            
        Returns:
            Data with added charts
        """
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # Create dictionary for charts
        charts = {}
        
        # Generate equity curve chart
        if 'equity_curve' in data and len(data['equity_curve']) > 0:
            equity_curve = data['equity_curve']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(equity_curve.index, equity_curve['equity'])
            ax.set_title('Equity Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            ax.grid(True)
            
            # Save figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for embedding in HTML
            charts['equity_curve'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            plt.close(fig)
            
        # Generate returns distribution chart
        if 'returns' in data and len(data['returns']) > 0:
            returns = data['returns']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(returns['return'], bins=50)
            ax.set_title('Returns Distribution')
            ax.set_xlabel('Return')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            
            # Save figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for embedding in HTML
            charts['returns_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            plt.close(fig)
            
        # Generate drawdown chart
        if 'equity_curve' in data and len(data['equity_curve']) > 0:
            equity_curve = data['equity_curve']
            
            # Calculate drawdown
            equity_values = equity_curve['equity']
            drawdowns = []
            peak = equity_values[0]
            
            for value in equity_values:
                if value > peak:
                    peak = value
                    drawdowns.append(0)
                else:
                    drawdown = (peak - value) / peak
                    drawdowns.append(drawdown)
                    
            equity_curve['drawdown'] = drawdowns
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(equity_curve.index, equity_curve['drawdown'] * 100)
            ax.set_title('Drawdown')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True)
            ax.invert_yaxis()
            
            # Save figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for embedding in HTML
            charts['drawdown'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            plt.close(fig)
            
        # Add charts to data
        data['charts'] = charts
        
        return data

### System Status Reports

class SystemStatusDataProvider:
    """Data provider for system status reports."""
    
    def __init__(self, system_metrics_service):
        """
        Initialize data provider.
        
        Args:
            system_metrics_service: Service for accessing system metrics
        """
        self.metrics_service = system_metrics_service
        
    def get_data(self, parameters):
        """
        Get system status data.
        
        Args:
            parameters: Parameters for data retrieval
            
        Returns:
            System status data
        """
        # Extract parameters
        start_time = parameters.get('start_time')
        end_time = parameters.get('end_time')
        include_components = parameters.get('include_components', True)
        
        # Get system metrics
        metrics = self.metrics_service.get_system_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Get component metrics if requested
        component_metrics = {}
        if include_components:
            component_ids = self.metrics_service.get_component_ids()
            
            for component_id in component_ids:
                component_metrics[component_id] = self.metrics_service.get_component_metrics(
                    component_id,
                    start_time=start_time,
                    end_time=end_time
                )
                
        # Calculate summary statistics
        summary = self._calculate_summary(metrics)
        
        # Combine data
        return {
            'system': {
                'name': 'ADMF Trading System',
                'environment': self.metrics_service.get_environment()
            },
            'summary': summary,
            'metrics': metrics,
            'components': component_metrics
        }
        
    def _calculate_summary(self, metrics):
        """Calculate summary statistics."""
        import numpy as np
        
        # Initialize summary
        summary = {}
        
        # Calculate CPU metrics
        if 'cpu' in metrics and len(metrics['cpu']) > 0:
            cpu_usage = np.array(metrics['cpu']['usage'])
            summary['cpu'] = {
                'avg_usage': cpu_usage.mean(),
                'max_usage': cpu_usage.max(),
                'min_usage': cpu_usage.min()
            }
            
        # Calculate memory metrics
        if 'memory' in metrics and len(metrics['memory']) > 0:
            memory_usage = np.array(metrics['memory']['usage_percent'])
            summary['memory'] = {
                'avg_usage': memory_usage.mean(),
                'max_usage': memory_usage.max(),
                'min_usage': memory_usage.min()
            }
            
        # Calculate disk metrics
        if 'disk' in metrics:
            summary['disk'] = {
                'read_bytes': metrics['disk'].get('read_bytes', 0),
                'write_bytes': metrics['disk'].get('write_bytes', 0)
            }
            
        # Calculate network metrics
        if 'network' in metrics:
            summary['network'] = {
                'received_bytes': metrics['network'].get('received_bytes', 0),
                'sent_bytes': metrics['network'].get('sent_bytes', 0)
            }
            
        # Calculate request metrics
        if 'requests' in metrics and len(metrics['requests']) > 0:
            request_times = np.array(metrics['requests']['time'])
            summary['requests'] = {
                'count': len(request_times),
                'avg_time': request_times.mean(),
                'p90_time': np.percentile(request_times, 90),
                'p99_time': np.percentile(request_times, 99)
            }
            
        return summary

class SystemStatusChartGenerator:
    """Chart generator for system status reports."""
    
    def process(self, data, parameters):
        """
        Generate charts for system status data.
        
        Args:
            data: System status data
            parameters: Processing parameters
            
        Returns:
            Data with added charts
        """
        import matplotlib.pyplot as plt
        import io
        import base64
        import numpy as np
        
        # Create dictionary for charts
        charts = {}
        
        # Generate CPU usage chart
        if 'metrics' in data and 'cpu' in data['metrics'] and len(data['metrics']['cpu']) > 0:
            cpu_metrics = data['metrics']['cpu']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(cpu_metrics.index, cpu_metrics['usage'])
            ax.set_title('CPU Usage')
            ax.set_xlabel('Time')
            ax.set_ylabel('Usage (%)')
            ax.set_ylim(0, 100)
            ax.grid(True)
            
            # Save figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for embedding in HTML
            charts['cpu_usage'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            plt.close(fig)
            
        # Generate memory usage chart
        if 'metrics' in data and 'memory' in data['metrics'] and len(data['metrics']['memory']) > 0:
            memory_metrics = data['metrics']['memory']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(memory_metrics.index, memory_metrics['usage_percent'])
            ax.set_title('Memory Usage')
            ax.set_xlabel('Time')
            ax.set_ylabel('Usage (%)')
            ax.set_ylim(0, 100)
            ax.grid(True)
            
            # Save figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for embedding in HTML
            charts['memory_usage'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            plt.close(fig)
            
        # Generate request time chart
        if 'metrics' in data and 'requests' in data['metrics'] and len(data['metrics']['requests']) > 0:
            request_metrics = data['metrics']['requests']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(request_metrics.index, request_metrics['time'])
            ax.set_title('Request Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Request Time (ms)')
            ax.grid(True)
            
            # Save figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for embedding in HTML
            charts['request_time'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            plt.close(fig)
            
        # Generate component CPU usage chart
        if 'components' in data and len(data['components']) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for component_id, metrics in data['components'].items():
                if 'cpu' in metrics and len(metrics['cpu']) > 0:
                    ax.plot(metrics['cpu'].index, metrics['cpu']['usage'], label=component_id)
                    
            ax.set_title('Component CPU Usage')
            ax.set_xlabel('Time')
            ax.set_ylabel('Usage (%)')
            ax.set_ylim(0, 100)
            ax.grid(True)
            ax.legend()
            
            # Save figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for embedding in HTML
            charts['component_cpu_usage'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            plt.close(fig)
            
        # Add charts to data
        data['charts'] = charts
        
        return data
```

### 4. Dynamic Report Templates

HTML templates for different report types:

```html
<!-- Strategy Performance Report Template -->
<!DOCTYPE html>
<html>
<head>
    <title>{{ metadata.report_name }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .header {
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 30px;
        }
        .metric-box {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            display: inline-block;
            width: 22%;
            margin-right: 2%;
            vertical-align: top;
        }
        .metric-box h3 {
            margin-top: 0;
            font-size: 16px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f5f5f5;
        }
        .chart-container {
            margin-bottom: 20px;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Strategy Performance Report</h1>
        <p>
            <strong>Strategy:</strong> {{ strategy.name }}<br>
            <strong>Period:</strong> {{ metadata.parameters.start_date }} to {{ metadata.parameters.end_date }}<br>
            <strong>Generated:</strong> {{ metadata.generated_at }}
        </p>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        
        <div class="metric-box">
            <h3>Total Return</h3>
            <div class="metric-value {% if summary.total_return >= 0 %}positive{% else %}negative{% endif %}">
                {{ "%.2f"|format(summary.total_return) }}%
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Annualized Return</h3>
            <div class="metric-value {% if summary.annualized_return >= 0 %}positive{% else %}negative{% endif %}">
                {{ "%.2f"|format(summary.annualized_return) }}%
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Sharpe Ratio</h3>
            <div class="metric-value">
                {{ "%.2f"|format(summary.sharpe_ratio) }}
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Max Drawdown</h3>
            <div class="metric-value negative">
                {{ "%.2f"|format(summary.max_drawdown) }}%
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Trade Count</h3>
            <div class="metric-value">
                {{ summary.trade_count }}
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Win Rate</h3>
            <div class="metric-value">
                {{ "%.2f"|format(summary.win_rate) }}%
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Profit Factor</h3>
            <div class="metric-value">
                {{ "%.2f"|format(summary.profit_factor) }}
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Volatility</h3>
            <div class="metric-value">
                {{ "%.2f"|format(summary.volatility) }}%
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Equity Curve</h2>
        
        {% if charts.equity_curve %}
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.equity_curve }}" width="100%">
        </div>
        {% else %}
        <p>No equity curve data available.</p>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Drawdown</h2>
        
        {% if charts.drawdown %}
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.drawdown }}" width="100%">
        </div>
        {% else %}
        <p>No drawdown data available.</p>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Returns Distribution</h2>
        
        {% if charts.returns_distribution %}
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.returns_distribution }}" width="100%">
        </div>
        {% else %}
        <p>No returns distribution data available.</p>
        {% endif %}
    </div>
    
    {% if trades and trades|length > 0 %}
    <div class="section">
        <h2>Recent Trades</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Size</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Return</th>
                </tr>
            </thead>
            <tbody>
                {% for trade in trades[-10:] %}
                <tr>
                    <td>{{ trade.date }}</td>
                    <td>{{ trade.symbol }}</td>
                    <td>{{ trade.side }}</td>
                    <td>{{ trade.size }}</td>
                    <td>{{ "%.2f"|format(trade.entry_price) }}</td>
                    <td>{{ "%.2f"|format(trade.exit_price) }}</td>
                    <td class="{% if trade.return >= 0 %}positive{% else %}negative{% endif %}">
                        {{ "%.2f"|format(trade.return * 100) }}%
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}
</body>
</html>
```

```html
<!-- System Status Report Template -->
<!DOCTYPE html>
<html>
<head>
    <title>{{ metadata.report_name }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .header {
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 30px;
        }
        .metric-box {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
            display: inline-block;
            width: 22%;
            margin-right: 2%;
            vertical-align: top;
        }
        .metric-box h3 {
            margin-top: 0;
            font-size: 16px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f5f5f5;
        }
        .chart-container {
            margin-bottom: 20px;
        }
        .healthy {
            color: green;
        }
        .warning {
            color: orange;
        }
        .critical {
            color: red;
        }
        .component-status {
            display: inline-block;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-healthy {
            background-color: green;
        }
        .status-warning {
            background-color: orange;
        }
        .status-critical {
            background-color: red;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>System Status Report</h1>
        <p>
            <strong>System:</strong> {{ system.name }}<br>
            <strong>Environment:</strong> {{ system.environment }}<br>
            <strong>Period:</strong> {{ metadata.parameters.start_time }} to {{ metadata.parameters.end_time }}<br>
            <strong>Generated:</strong> {{ metadata.generated_at }}
        </p>
    </div>
    
    <div class="section">
        <h2>System Health Summary</h2>
        
        <div class="metric-box">
            <h3>Average CPU Usage</h3>
            <div class="metric-value {% if summary.cpu.avg_usage < 70 %}healthy{% elif summary.cpu.avg_usage < 90 %}warning{% else %}critical{% endif %}">
                {{ "%.1f"|format(summary.cpu.avg_usage) }}%
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Peak CPU Usage</h3>
            <div class="metric-value {% if summary.cpu.max_usage < 90 %}healthy{% elif summary.cpu.max_usage < 95 %}warning{% else %}critical{% endif %}">
                {{ "%.1f"|format(summary.cpu.max_usage) }}%
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Average Memory Usage</h3>
            <div class="metric-value {% if summary.memory.avg_usage < 70 %}healthy{% elif summary.memory.avg_usage < 90 %}warning{% else %}critical{% endif %}">
                {{ "%.1f"|format(summary.memory.avg_usage) }}%
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Peak Memory Usage</h3>
            <div class="metric-value {% if summary.memory.max_usage < 90 %}healthy{% elif summary.memory.max_usage < 95 %}warning{% else %}critical{% endif %}">
                {{ "%.1f"|format(summary.memory.max_usage) }}%
            </div>
        </div>
        
        {% if summary.requests %}
        <div class="metric-box">
            <h3>Request Count</h3>
            <div class="metric-value">
                {{ summary.requests.count }}
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Average Request Time</h3>
            <div class="metric-value {% if summary.requests.avg_time < 100 %}healthy{% elif summary.requests.avg_time < 500 %}warning{% else %}critical{% endif %}">
                {{ "%.1f"|format(summary.requests.avg_time) }} ms
            </div>
        </div>
        
        <div class="metric-box">
            <h3>90th Percentile Time</h3>
            <div class="metric-value {% if summary.requests.p90_time < 200 %}healthy{% elif summary.requests.p90_time < 1000 %}warning{% else %}critical{% endif %}">
                {{ "%.1f"|format(summary.requests.p90_time) }} ms
            </div>
        </div>
        
        <div class="metric-box">
            <h3>99th Percentile Time</h3>
            <div class="metric-value {% if summary.requests.p99_time < 500 %}healthy{% elif summary.requests.p99_time < 2000 %}warning{% else %}critical{% endif %}">
                {{ "%.1f"|format(summary.requests.p99_time) }} ms
            </div>
        </div>
        {% endif %}
        
        {% if summary.disk %}
        <div class="metric-box">
            <h3>Disk Read</h3>
            <div class="metric-value">
                {{ (summary.disk.read_bytes / 1024 / 1024)|round(1) }} MB
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Disk Write</h3>
            <div class="metric-value">
                {{ (summary.disk.write_bytes / 1024 / 1024)|round(1) }} MB
            </div>
        </div>
        {% endif %}
        
        {% if summary.network %}
        <div class="metric-box">
            <h3>Network Received</h3>
            <div class="metric-value">
                {{ (summary.network.received_bytes / 1024 / 1024)|round(1) }} MB
            </div>
        </div>
        
        <div class="metric-box">
            <h3>Network Sent</h3>
            <div class="metric-value">
                {{ (summary.network.sent_bytes / 1024 / 1024)|round(1) }} MB
            </div>
        </div>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>CPU Usage</h2>
        
        {% if charts.cpu_usage %}
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.cpu_usage }}" width="100%">
        </div>
        {% else %}
        <p>No CPU usage data available.</p>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Memory Usage</h2>
        
        {% if charts.memory_usage %}
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.memory_usage }}" width="100%">
        </div>
        {% else %}
        <p>No memory usage data available.</p>
        {% endif %}
    </div>
    
    {% if charts.request_time %}
    <div class="section">
        <h2>Request Time</h2>
        
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.request_time }}" width="100%">
        </div>
    </div>
    {% endif %}
    
    {% if components and components|length > 0 %}
    <div class="section">
        <h2>Component Status</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>CPU Usage (Avg)</th>
                    <th>Memory Usage (Avg)</th>
                    <th>Request Count</th>
                    <th>Avg Response Time</th>
                </tr>
            </thead>
            <tbody>
                {% for component_id, metrics in components.items() %}
                <tr>
                    <td>{{ component_id }}</td>
                    <td>
                        {% set status = metrics.status if metrics.status is defined else 'healthy' %}
                        <span class="component-status status-{{ status }}"></span>
                        {{ status|capitalize }}
                    </td>
                    <td>
                        {% if metrics.cpu and metrics.cpu.usage is defined %}
                        {{ "%.1f"|format(metrics.cpu.usage.mean()) }}%
                        {% else %}
                        N/A
                        {% endif %}
                    </td>
                    <td>
                        {% if metrics.memory and metrics.memory.usage_percent is defined %}
                        {{ "%.1f"|format(metrics.memory.usage_percent.mean()) }}%
                        {% else %}
                        N/A
                        {% endif %}
                    </td>
                    <td>
                        {% if metrics.requests and metrics.requests.count is defined %}
                        {{ metrics.requests.count }}
                        {% else %}
                        N/A
                        {% endif %}
                    </td>
                    <td>
                        {% if metrics.requests and metrics.requests.time is defined %}
                        {{ "%.1f"|format(metrics.requests.time.mean()) }} ms
                        {% else %}
                        N/A
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    {% if charts.component_cpu_usage %}
    <div class="section">
        <h2>Component CPU Usage</h2>
        
        <div class="chart-container">
            <img src="data:image/png;base64,{{ charts.component_cpu_usage }}" width="100%">
        </div>
    </div>
    {% endif %}
    {% endif %}
</body>
</html>
```

## Implementation Strategy

The reporting system implementation involves several components:

### 1. Core Framework

1. **Report Templates**:
   - HTML templates for different report types
   - CSS styling for consistent look and feel
   - Template variables for dynamic content

2. **Report Generator**:
   - Template rendering engine
   - Data collection and processing
   - Format conversion (HTML, PDF, CSV, etc.)

3. **Scheduler**:
   - Schedule definitions (daily, weekly, etc.)
   - Job execution
   - Error handling and retries

### 2. Data Providers

1. **Strategy Metrics Provider**:
   - Performance data (returns, drawdowns)
   - Risk metrics (volatility, Sharpe ratio)
   - Trade data (entries, exits, sizes)

2. **System Metrics Provider**:
   - CPU and memory usage
   - Disk and network I/O
   - Request counts and timings

3. **Operational Metrics Provider**:
   - Error rates and types
   - System health indicators
   - Dependency statuses

### 3. Output Formats

1. **HTML Reports**:
   - Interactive web-based reports
   - Embedded charts and graphics
   - Styled tables and metrics

2. **PDF Reports**:
   - Publication-quality printable reports
   - Corporate branding and styling
   - Document headers and footers

3. **Data Exports**:
   - CSV for data analysis
   - JSON for programmatic access
   - Excel for spreadsheet users

## Best Practices

### 1. Report Design

- **Prioritize Key Information**:
  ```python
  # Clearly identify the most important metrics
  report_definition.add_post_processor(lambda data, params: {
      **data,
      'highlight_metrics': {
          'total_return': data['summary']['total_return'],
          'sharpe_ratio': data['summary']['sharpe_ratio'],
          'max_drawdown': data['summary']['max_drawdown']
      }
  })
  ```

- **Use Consistent Formatting**:
  ```html
  <!-- Use consistent formatting for all numeric values -->
  <div class="metric-value {% if value >= 0 %}positive{% else %}negative{% endif %}">
      {{ "%.2f"|format(value) }}%
  </div>
  ```

- **Include Context and Comparisons**:
  ```python
  # Add benchmark comparisons to reports
  def add_benchmark_comparison(data, params):
      benchmark_id = params.get('benchmark_id', 'SPY')
      benchmark_data = get_benchmark_data(benchmark_id, 
                                         params.get('start_date'),
                                         params.get('end_date'))
      
      # Calculate relative performance
      relative_return = data['summary']['total_return'] - benchmark_data['total_return']
      
      data['benchmark'] = {
          'id': benchmark_id,
          'return': benchmark_data['total_return'],
          'relative_return': relative_return
      }
      
      return data
      
  report_definition.add_post_processor(add_benchmark_comparison)
  ```

### 2. Data Processing

- **Preprocess Heavy Calculations**:
  ```python
  # Preprocess data-intensive calculations
  class PerformanceMetricsProcessor:
      def process(self, data, parameters):
          # Calculate metrics only once
          if 'equity_curve' in data and len(data['equity_curve']) > 0:
              equity = data['equity_curve']['equity']
              
              # Calculate drawdown series
              drawdown_series = self._calculate_drawdown_series(equity)
              data['drawdown_series'] = drawdown_series
              
              # Calculate rolling metrics
              window = parameters.get('rolling_window', 20)
              data['rolling_metrics'] = self._calculate_rolling_metrics(
                  data['returns'], window)
                  
          return data
      
      def _calculate_drawdown_series(self, equity):
          # Implementation of drawdown calculation
          pass
          
      def _calculate_rolling_metrics(self, returns, window):
          # Implementation of rolling metrics calculation
          pass
  ```

- **Cache Expensive Operations**:
  ```python
  # Cache expensive chart generation
  class CachingChartGenerator:
      def __init__(self, cache_expiry=300):  # 5 minutes
          self.cache = {}
          self.cache_expiry = cache_expiry
          
      def process(self, data, parameters):
          # Generate cache key
          cache_key = self._generate_cache_key(data, parameters)
          
          # Check cache
          current_time = time.time()
          if cache_key in self.cache:
              cached_data, timestamp = self.cache[cache_key]
              if current_time - timestamp < self.cache_expiry:
                  return cached_data
                  
          # Generate charts
          processed_data = self._generate_charts(data, parameters)
          
          # Store in cache
          self.cache[cache_key] = (processed_data, current_time)
          
          return processed_data
          
      def _generate_cache_key(self, data, parameters):
          # Implementation of cache key generation
          pass
          
      def _generate_charts(self, data, parameters):
          # Implementation of chart generation
          pass
  ```

- **Paginate Large Datasets**:
  ```python
  # Paginate large trade lists
  def paginate_trades(data, parameters):
      page_size = parameters.get('page_size', 50)
      page = parameters.get('page', 1)
      
      if 'trades' in data and len(data['trades']) > page_size:
          # Calculate start and end indices
          start_idx = (page - 1) * page_size
          end_idx = start_idx + page_size
          
          # Paginate trades
          total_trades = len(data['trades'])
          data['trades'] = data['trades'][start_idx:end_idx]
          
          # Add pagination metadata
          data['pagination'] = {
              'page': page,
              'page_size': page_size,
              'total_trades': total_trades,
              'total_pages': (total_trades + page_size - 1) // page_size
          }
          
      return data
  ```

### 3. Scheduling and Delivery

- **Adaptive Scheduling**:
  ```python
  # Adapt report schedule to trading hours
  def create_market_hours_schedule():
      # Get market calendar
      market_calendar = get_market_calendar()
      
      for date in market_calendar.get_trading_days(start_date, end_date):
          # Create pre-market report
          scheduler.add_schedule(
              'daily_strategy_report',
              'daily',
              parameters={
                  'time': '08:45',  # 15 minutes before market open
                  'title': 'Pre-Market Strategy Report'
              },
              delivery_method=ReportDeliveryMethod.EMAIL,
              delivery_parameters={
                  'recipients': ['traders@example.com']
              }
          )
          
          # Create post-market report
          scheduler.add_schedule(
              'daily_strategy_report',
              'daily',
              parameters={
                  'time': '16:15',  # 15 minutes after market close
                  'title': 'Post-Market Strategy Report'
              },
              delivery_method=ReportDeliveryMethod.EMAIL,
              delivery_parameters={
                  'recipients': ['traders@example.com', 'management@example.com']
              }
          )
  ```

- **Smart Delivery**:
  ```python
  # Deliver reports based on content significance
  class SmartDeliveryProcessor:
      def process(self, data, parameters):
          # Analyze report for significant events
          significant_events = self._detect_significant_events(data)
          
          # Add delivery instructions
          if significant_events:
              data['delivery'] = {
                  'priority': 'high',
                  'additional_recipients': ['risk@example.com', 'alerts@example.com'],
                  'notification_text': f"ALERT: {len(significant_events)} significant events detected"
              }
          else:
              data['delivery'] = {
                  'priority': 'normal'
              }
              
          return data
          
      def _detect_significant_events(self, data):
          # Implementation of significant event detection
          events = []
          
          # Check for large drawdowns
          if 'summary' in data and 'max_drawdown' in data['summary']:
              if data['summary']['max_drawdown'] > 10:  # 10% drawdown
                  events.append({
                      'type': 'drawdown',
                      'value': data['summary']['max_drawdown'],
                      'threshold': 10
                  })
                  
          # Check for unusual volatility
          if 'summary' in data and 'volatility' in data['summary']:
              if data['summary']['volatility'] > 25:  # 25% annualized volatility
                  events.append({
                      'type': 'volatility',
                      'value': data['summary']['volatility'],
                      'threshold': 25
                  })
                  
          return events
  ```

- **Error Handling**:
  ```python
  # Graceful handling of report generation errors
  def safe_report_generation(report_name, parameters):
      try:
          # Generate report
          report_content = report_generator.create_report(report_name, parameters)
          return report_content
      except Exception as e:
          logging.error(f"Error generating report {report_name}: {e}")
          
          # Generate error report
          error_template = report_generator.templates.get('error_report')
          if error_template:
              return error_template.render({
                  'error': str(e),
                  'report_name': report_name,
                  'parameters': parameters,
                  'timestamp': datetime.datetime.now(),
                  'traceback': traceback.format_exc()
              })
          else:
              # Simple error message
              return f"Error generating report {report_name}: {e}"
  ```

## Conclusion

The Reporting System provides a comprehensive framework for transforming raw data and metrics into actionable insights. By combining flexible templates, scheduled delivery, and domain-specific data providers, it enables users to monitor trading strategies, system performance, and operational health effectively.

Key benefits include:

1. **Automated Insights**: Regular reports delivered on schedule without manual intervention

2. **Consistent Formatting**: Standardized templates ensure consistent presentation across reports

3. **Multiple Formats**: Support for HTML, PDF, CSV, and other formats to meet different user needs

4. **Extensibility**: Easy addition of new report types, data providers, and delivery methods

5. **Performance Optimization**: Efficient data processing and caching for responsive report generation

These capabilities ensure that all users have access to the information they need, in the format they prefer, at the time when it's most valuable.