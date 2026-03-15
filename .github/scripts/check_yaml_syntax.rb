#!/usr/bin/env ruby

require "yaml"

failed = false

ARGV.each do |path|
  begin
    YAML.load_file(path)
  rescue StandardError => e
    failed = true
    warn "#{path}: #{e.message}"
  end
end

exit(failed ? 1 : 0)
