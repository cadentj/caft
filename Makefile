all:
	@mkdir -p docs
	cp -r src/* docs/

clean:
	rm -rf docs

.PHONY: all cleanw